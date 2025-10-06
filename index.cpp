#include <unordered_map>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

#include <torch/optim/schedulers/lr_scheduler.h>
#include <torch/data/transforms/stack.h>
#include <torch/data/datasets/tensor.h>
#include <torch/serialize/archive.h>
#include <torch/torch.h>

// Data settings
static inline const int CONTEXT_WINDOW = 192;
static inline const int BATCH_SIZE = 96;

// Transformer settings
static inline const int EPOCHS = 250;
static inline const int D_MODEL = 256;
static inline const int DFF = 1024;
static inline const int N_HEADS = 8;
static inline const int N_BLOCKS = 6;

// Training settings
static inline const int VALIDATION_STEPS = 500;
static inline const int SAVE_STEPS = 1000;
static inline const int GRAD_ACCUM = 2;

static inline const int STRIDE = CONTEXT_WINDOW / 8;

torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);

struct PairDataset : torch::data::datasets::Dataset<PairDataset> {
    torch::Tensor X;
    torch::Tensor Y;

    PairDataset(torch::Tensor _X, torch::Tensor _Y) : X(std::move(_X)), Y(std::move(_Y)) {

    }

    torch::data::Example<> get(size_t i) override {
        return {X[i], Y[i]};
    }

    torch::optional<size_t> size() const override {
        return X.size(0);
    }
};

struct WarmupCosine {
    int64_t warmup;
    int64_t total;
    int64_t step;

    double baseLr;
    double minLr;

    WarmupCosine(int64_t warmupSteps, int64_t totalSteps, double _baseLr, double _minLr) {
        warmup = warmupSteps;
        total = totalSteps;

        baseLr = _baseLr;
        minLr = _minLr;
        
        step = 0;
    }

    double returnLearningRate() const {
        if (step < warmup) {
            double w = double(step) / std::max<int64_t>(1, warmup);
            return minLr + (baseLr - minLr) * w;
        }

        double t = double(step - warmup) / std::max<int64_t>(1, total - warmup);
        double cosv = 0.5 * (1 + std::cos(M_PI * t));

        return minLr + (baseLr - minLr) * cosv;
    }

    void stepOptimizer(torch::optim::Optimizer& optimizer) {
        double lr = returnLearningRate();

        for (auto&group: optimizer.param_groups()) {
            auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
            options.set_lr(lr);
        }
        step++;
    }
};

class Codec {
    private:
    std::string _readFile(std::string file) {
        std::ifstream inputFile(file);

        std::string result = "";
        std::string line = "";

        while (std::getline(inputFile, line)) {
            result += line;
            result += '\n';
        }

        inputFile.close();
        return result;
    }

    public:
    std::unordered_map<char, int> characterToId;
    std::unordered_map<int, char> idToCharacter;

    std::vector<int> dataset;
    torch::Tensor allBatches;

    int size;

    Codec(std::string file) {
        std::string fileContents = _readFile(file);

        int id = 1;
        for (char c: fileContents) {
            if (characterToId.find(c) != characterToId.end()) {
                continue;
            }

            characterToId.insert(std::pair<char, int>(c, id));
            idToCharacter.insert(std::pair<int, char>(id, c));

            id++;
        }

        size = characterToId.size() + 1;
        dataset = encode(fileContents);

        allBatches = splitSamples(dataset);
    }

    std::vector<int> encode(std::string input) {
        std::vector<int> tokens;

        for (char c: input) {
            tokens.push_back(characterToId.find(c) != characterToId.end() ? characterToId.at(c) : 0);
        }

        return tokens;
    }

    std::string decode(std::vector<int>& tokens) {
        std::string result = "";

        for (int id: tokens) {
            result += idToCharacter.find(id) != idToCharacter.end() ? idToCharacter.at(id) : '?';
        }

        return result;
    }

    torch::Tensor splitSamples(const std::vector<int>& encoded) {
        std::vector<int64_t> encoded64(encoded.begin(), encoded.end());
        torch::Tensor encodedTensor = torch::tensor(encoded64, torch::TensorOptions().dtype(torch::kLong));

        int N = encodedTensor.size(0);
        std::vector<torch::Tensor> batches;

        batches.reserve((N > CONTEXT_WINDOW) ? (N - CONTEXT_WINDOW) / (CONTEXT_WINDOW + 1) : 0);

        for (int64_t i = 0; i  < N - CONTEXT_WINDOW; i += STRIDE) { //i += 8
            auto X = encodedTensor.slice(0, i, i + CONTEXT_WINDOW);
            auto Y = encodedTensor.slice(0, i + 1, i + CONTEXT_WINDOW + 1);

            std::vector<torch::Tensor> pair;
            pair.reserve(2);
            pair.push_back(X);
            pair.push_back(Y);

            batches.push_back(torch::stack(pair));
        }

        return torch::stack(batches);
    }
};

class AttentionHead {
    public:
    int dmodel;
    int dk;

    torch::Tensor qw;
    torch::Tensor kw;
    torch::Tensor vw;

    AttentionHead(int _dmodel, int _dk) {
        dmodel = _dmodel;
        dk = _dk;

        float scale = 1.0f / std::sqrt((float) dmodel);

        qw = scale * torch::randn({dmodel, dk}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        qw.set_requires_grad(true);

        kw = scale * torch::randn({dmodel, dk}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        kw.set_requires_grad(true);

        vw = scale * torch::randn({dmodel, dk}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        vw.set_requires_grad(true);
    }

    torch::Tensor forward(torch::Tensor inputEmbedding) {
        torch::Tensor q = torch::matmul(inputEmbedding, qw);
        torch::Tensor k = torch::matmul(inputEmbedding, kw);
        torch::Tensor v = torch::matmul(inputEmbedding, vw);

        torch::Tensor scores = torch::matmul(q, k.transpose(1, 2)) * (1 / std::sqrt((double) dk));
        int64_t S = scores.size(-1);

        torch::Tensor mask = torch::ones({S, S}, torch::TensorOptions().dtype(torch::kBool).device(scores.device())).triu(1);
        scores = scores.masked_fill(mask, -1e9f);

        torch::Tensor attention = torch::softmax(scores, -1);
        return torch::matmul(attention, v);
    }

    std::vector<torch::Tensor> parameters() {
        std::vector<torch::Tensor> result = {
            qw,
            kw,
            vw
        };

        return result;
    }
};

class MultiHeadAttention {
    public:
    std::vector<AttentionHead> heads;

    int numberOfHeads;
    int dmodel;
    int dk;

    torch::Tensor ow;
    torch::Tensor ob;
    
    MultiHeadAttention(int _numberOfHeads = 8, int _dmodel = 128) {
        numberOfHeads = _numberOfHeads;
        dmodel = _dmodel;
        dk = floor(dmodel / numberOfHeads);

        for (int i = 0; i < numberOfHeads; i++) {
            AttentionHead h(dmodel, dk);
            heads.push_back(h);
        }

        ow = 0.02 * torch::randn({numberOfHeads * dk, dmodel}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        ob = torch::zeros({dmodel}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

        ow.set_requires_grad(true);
        ob.set_requires_grad(true);
    }

    torch::Tensor forward(torch::Tensor h) {
        std::vector<torch::Tensor> outputs;

        for (AttentionHead& head: heads) {
            outputs.push_back(head.forward(h));
        }

        torch::Tensor concat = torch::cat(outputs, -1);

        return torch::matmul(concat, ow) + ob;
    }

    std::vector<torch::Tensor> parameters() {
        std::vector<torch::Tensor> result;

        for (int i = 0; i < heads.size(); i++) {
            std::vector<torch::Tensor> parameters = heads[i].parameters();

            for (int j = 0; j < parameters.size(); j++) {
                result.push_back(parameters[j]);
            }
        }

        result.push_back(ow);
        result.push_back(ob);

        return result;
    }
};

class FeedForwardNetwork {
    public:
    int dmodel;
    int dff;

    torch::Tensor w1, b1, w2, b2;

    FeedForwardNetwork(int _dmodel = 128, int _dff = 512) {
        dmodel = _dmodel;
        dff = _dff;
        w1 = (1.0f / std::sqrt((double) dmodel)) * torch::randn({dmodel, dff}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        w1.set_requires_grad(true);

        b1 = torch::zeros({dff}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        b1.set_requires_grad(true);

        w2 = (1.0f / std::sqrt((double) dff)) * torch::randn({dff, dmodel}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        w2.set_requires_grad(true);

        b2 = torch::zeros({dmodel}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        b2.set_requires_grad(true);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::matmul(x, w1) + b1;
        x = torch::gelu(x);
        x = torch::matmul(x, w2) + b2;

        return x;
    }

    std::vector<torch::Tensor> parameters() {
        std::vector<torch::Tensor> result = {
            w1,
            b1,
            w2,
            b2
        };
        return result;
    }
};

class TransformerBlock {
    public:
    int numberOfHeads;
    int dmodel;
    int dff;

    std::unique_ptr<FeedForwardNetwork> feedForwardNetwork;
    std::unique_ptr<MultiHeadAttention> attention;

    torch::nn::LayerNorm feedForwardLayerNorm{nullptr};
    torch::nn::LayerNorm attentionLayerNorm{nullptr};
    torch::nn::Dropout dropout{nullptr};

    TransformerBlock(int _dmodel = 128, int _dff = 512, int _numberOfHeads = 8) {
        numberOfHeads = _numberOfHeads;
        dmodel = _dmodel;
        dff = _dff;

        feedForwardNetwork = std::make_unique<FeedForwardNetwork>(dmodel, dff);
        attention = std::make_unique<MultiHeadAttention>(numberOfHeads, dmodel);

        feedForwardLayerNorm = torch::nn::LayerNorm(
            torch::nn::LayerNormOptions(std::vector<int64_t>{dmodel})
        );
        attentionLayerNorm = torch::nn::LayerNorm(
            torch::nn::LayerNormOptions(std::vector<int64_t>{dmodel})
        );

        dropout = torch::nn::Dropout(0.10f);

        feedForwardLayerNorm.get()->to(device);
        attentionLayerNorm.get()->to(device);
        dropout.get()->to(device);
    }

    torch::Tensor forward(torch::Tensor h) {
        torch::Tensor h1 = attentionLayerNorm.get()->forward(h);
        torch::Tensor attentionOut = attention.get()->forward(h1);
        h = h + dropout.get()->forward(attentionOut);

        torch::Tensor h2 = feedForwardLayerNorm.get()->forward(h);
        torch::Tensor feedForwardOut = feedForwardNetwork.get()->forward(h2);
        h = h + dropout.get()->forward(feedForwardOut);

        return h;
    }

    std::vector<torch::Tensor> parameters() {
        std::vector<torch::Tensor> result;

        std::vector<torch::Tensor> attentionParams = attention.get()->parameters();
        for (int i = 0; i < attentionParams.size(); i++) {
            result.push_back(attentionParams[i]);
        }

        std::vector<torch::Tensor> feedForwardParams = feedForwardNetwork.get()->parameters();
        for (int i = 0; i < feedForwardParams.size(); i++) {
            result.push_back(feedForwardParams[i]);
        }
        
        std::vector<torch::Tensor> attentionNormParams = attentionLayerNorm.get()->parameters();
        for (int i = 0; i < attentionNormParams.size(); i++) {
            result.push_back(attentionNormParams[i]);
        }

        std::vector<torch::Tensor> feedForwardLayerParams = feedForwardLayerNorm.get()->parameters();
        for (int i = 0; i < feedForwardLayerParams.size(); i++) {
            result.push_back(feedForwardLayerParams[i]);
        }

        return result;
    }
};

class Transformer {
    public:
    int numberOfBlocks;
    int numberOfHeads;
    int dmodel;
    int dff;

    std::vector<TransformerBlock> blocks;
    torch::Tensor embedding;
    torch::Tensor projB;

    torch::Tensor positionEmbed;

    std::unique_ptr<torch::optim::AdamW> optimizer;
    std::unique_ptr<WarmupCosine> scheduler;
    std::unique_ptr<torch::nn::CrossEntropyLoss> lossFunction;

    std::unique_ptr<Codec> codec;

    torch::nn::LayerNorm outNorm{nullptr};

    static torch::Tensor buildPositionalEncoding(int maxLength, int dmodel) {
        torch::Tensor position = torch::arange(0, maxLength, torch::kFloat32).to(device).unsqueeze(1);
        torch::Tensor i = torch::arange(0, dmodel, 2, torch::kFloat32).to(device).unsqueeze(0);
        torch::Tensor division = torch::exp(-std::log(10000.0f) * (i / dmodel));

        torch::Tensor encoded = torch::zeros({maxLength, dmodel}, torch::kFloat32).to(device);
        encoded.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)}, torch::sin(position * division));
        encoded.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)}, torch::cos(position * division));

        return encoded;
    }

    Transformer(int _dmodel = 128, int _dff = 512, int _numberOfHeads = 8, int _numberOfBlocks = 10) {
        numberOfBlocks = _numberOfBlocks;
        numberOfHeads = _numberOfHeads;
        dmodel = _dmodel;
        dff = _dff;

        codec = std::make_unique<Codec>("dataset.txt");

        outNorm = torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{dmodel}));
        outNorm.get()->to(device);

        for (int i = 0; i < numberOfBlocks; i++) {
            blocks.push_back(TransformerBlock(dmodel, dff, numberOfHeads));
        }

        embedding = 0.02f * torch::randn({codec.get()->size, dmodel}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        embedding.set_requires_grad(true);

        projB = torch::zeros({codec.get()->size}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        projB.set_requires_grad(true);

        positionEmbed = buildPositionalEncoding(CONTEXT_WINDOW, dmodel);

        torch::optim::AdamWOptions adamOptions;
        adamOptions.set_lr(5e-4);
        adamOptions.betas(std::make_tuple(0.9, 0.95));
        adamOptions.weight_decay(0.01);

        optimizer = std::make_unique<torch::optim::AdamW>(parameters(), adamOptions);
        scheduler = std::make_unique<WarmupCosine>(0, 1, 5e-4, 1e-4);

        lossFunction = std::make_unique<torch::nn::CrossEntropyLoss>(
            torch::nn::CrossEntropyLossOptions().label_smoothing(0.1).reduction(torch::kMean)
        );
    }

    torch::Tensor forward(torch::Tensor X) {
        X = X.to(device, torch::kLong);

        torch::Tensor embeddedX = torch::embedding(embedding, X);
        embeddedX = embeddedX * std::sqrt((float) dmodel);

        const int64_t S = embeddedX.size(1);
        auto dataType = embeddedX.dtype();

        embeddedX = embeddedX + positionEmbed.index({torch::indexing::Slice(0, S)}).unsqueeze(0);

        for (auto& block: blocks) {
            embeddedX = block.forward(embeddedX);
        }

        embeddedX = outNorm.get()->forward(embeddedX);

        float scale = std::sqrt((float) dmodel);
        torch::Tensor logits = (torch::matmul(embeddedX, embedding.transpose(0, 1))) + projB;
        return logits;
    }

    double evaluate(auto loader) {
        torch::NoGradGuard guard;

        double lossCount = 0.0;
        size_t batches = 0;

        for (auto batch: *loader->get()) {
            auto Xs = batch.data.to(device);
            auto Ys = batch.target.to(device);

            torch::Tensor logits = forward(Xs).to(torch::kFloat32);

            auto logits2d = logits.flatten(0, -2);
            auto target1d = Ys.reshape({-1}).to(torch::kLong);

            torch::Tensor loss = lossFunction.get()->get()->forward(logits2d, target1d);
            lossCount += loss.item<double>();
            batches++;
        }

        lossCount = batches ? (lossCount / static_cast<double>(batches)) : 0.0;
        std::cout << "Evaluation Loss: " << lossCount << '\n';

        return lossCount;
    }

    torch::Tensor trainStep(torch::Tensor X, torch::Tensor Y) {
        torch::Tensor logits = forward(X).to(torch::kFloat32);
        
        int64_t V = logits.size(-1);

        torch::Tensor logits2d = logits.flatten(0, -2);
        torch::Tensor target1d = Y.reshape({-1}).to(torch::kLong);

        return lossFunction.get()->get()->forward(logits2d, target1d);
    }

    void toggleDropout(bool toggle) {
        for (auto& b: blocks) {
            b.dropout.get()->train(toggle);
        }
    }

    void train(int epochs = 100, int validationSteps = 50, int saveSteps = 50, int gradientAccumulationSteps = 32) {
        toggleDropout(true);

        torch::Tensor X = codec.get()->allBatches.select(1, 0).contiguous();
        torch::Tensor Y = codec.get()->allBatches.select(1, 1).to(torch::kLong).contiguous();

        auto trainData = PairDataset(X, Y).map(torch::data::transforms::Stack<>());
        auto trainLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(trainData),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2)
        );

        auto evalData = PairDataset(X, Y).map(torch::data::transforms::Stack<>());
        auto evalLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(evalData),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2)
        );

        optimizer.get()->zero_grad();

        size_t totalSamples = X.size(0);
        size_t batchesPerEpoch = (totalSamples + BATCH_SIZE - 1) / BATCH_SIZE;

        int64_t stepsPerEpoch = (batchesPerEpoch + gradientAccumulationSteps - 1) / gradientAccumulationSteps;
        scheduler.get()->step = 0;
        scheduler.get()->total = (int64_t) epochs * stepsPerEpoch;
        scheduler.get()->warmup = std::min<int64_t>(1000, scheduler.get()->total / 10);

        long long microStepCounter = 0;
        long long stepsForValidation = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            size_t b = 0;

            for (auto& batch: *trainLoader.get()) {
                b++;
                microStepCounter++;
                stepsForValidation++;

                auto Xs = batch.data.to(device);
                auto Ys = batch.target.to(device);

                torch::Tensor loss = trainStep(Xs, Ys) / static_cast<double>(gradientAccumulationSteps);
                loss.backward();

                bool lastBatchOfEpoch = (b == batchesPerEpoch);
                bool doStep = (microStepCounter % gradientAccumulationSteps == 0) || lastBatchOfEpoch;

                if (doStep) {
                    torch::nn::utils::clip_grad_norm_(parameters(), 1.0);
                    optimizer.get()->step();
                    optimizer.get()->zero_grad();
                    scheduler.get()->stepOptimizer(*optimizer.get());

                    float scaledLoss = loss.item<float>() * static_cast<float>(gradientAccumulationSteps);
                    std::cout << "Epoch: " << (epoch + 1) << ", Micro: " << microStepCounter << ", Loss: " << scaledLoss << '\n';

                    if (saveSteps > 0 && (microStepCounter % saveSteps) == 0) {
                        save("model.pt", true);
                    }
                }

                if (stepsForValidation % validationSteps == 0) {
                    toggleDropout(false);
                    evaluate(&evalLoader);
                    toggleDropout(true);
                }
            }

            toggleDropout(false);
            evaluate(&evalLoader);
            toggleDropout(true);
        }

        save("model.pt", true);
    }

    std::vector<torch::Tensor> parameters() {
        std::vector<torch::Tensor> result;

        result.push_back(embedding);
        result.push_back(projB);

        for (int i = 0; i < blocks.size(); i++) {
            TransformerBlock& currentBlock = blocks[i];
            std::vector<torch::Tensor> blockParams = currentBlock.parameters();

            for (int j = 0; j < blockParams.size(); j++) {
                result.push_back(blockParams[j]);
            }
        }
       
        return result;
    }

    std::string generate(std::string input, int steps = 200) {
        const float temperature = 0.4f;
        const double topp = 0.90;
        const int topk = 50;
        const float repetitionPenalty = 1.02f;

        toggleDropout(false);
        torch::NoGradGuard guard;

        std::vector<int> tokens = codec.get()->encode(input);
        std::vector<int64_t> tokens64(tokens.begin(), tokens.end());
        
        if (tokens64.empty()) {
            tokens64.push_back(0);
        }

        std::vector<int64_t> seen;
        seen.reserve(tokens64.size() + steps);

        for (int i = 0; i < steps; i++) {
            int64_t contextLength = std::min<int64_t>(tokens64.size(), CONTEXT_WINDOW);
            int64_t offset = static_cast<int64_t>(tokens64.size()) - contextLength;

            torch::Tensor context = torch::tensor(
                std::vector<int64_t>(tokens64.begin() + offset, tokens64.end()),
                torch::TensorOptions().dtype(torch::kLong)
            ).unsqueeze(0).to(device);

            torch::Tensor logits = forward(context).to(torch::kFloat32);
            torch::Tensor lastLogits = logits.select(1, logits.size(1) - 1).select(0, 0);

            lastLogits = lastLogits / temperature;
            torch::Tensor maxv = std::get<0>(lastLogits.max(0, false));
            lastLogits = lastLogits - maxv;

            if (!seen.empty()) {
                auto ids = torch::tensor(
                    seen,
                    torch::TensorOptions().dtype(torch::kLong).device(lastLogits.device())
                );
                ids = ids.clamp(0, lastLogits.size(0) - 1);

                torch::Tensor values = lastLogits.index({ids});
                torch::Tensor penalized = torch::where(values > 0, values / repetitionPenalty, values * repetitionPenalty);
                lastLogits.index_put_({ids}, penalized);
            }

            if (topk > 0 && topk < lastLogits.size(0)) {
                auto tk = lastLogits.topk(topk);
                torch::Tensor kth = std::get<0>(tk).index({topk - 1});
                lastLogits = lastLogits.masked_fill(lastLogits < kth, -std::numeric_limits<float>::infinity());
            }

            torch::Tensor probabilities = torch::softmax(lastLogits, 0);
            auto sorted = probabilities.sort(0, true);
            torch::Tensor sortedProbabilities = std::get<0>(sorted);
            torch::Tensor sortedIndex = std::get<1>(sorted);

            torch::Tensor cdf = sortedProbabilities.cumsum(0);
            torch::Tensor cutoffMask = cdf > topp;

            if (cutoffMask.numel() > 0) {
                cutoffMask.index_put_({0}, false);
            }

            torch::Tensor toMask = sortedIndex.index({cutoffMask});

            if (toMask.numel() > 0) {
                lastLogits.index_put_({toMask}, -std::numeric_limits<float>::infinity());
            }

            probabilities = torch::softmax(lastLogits, 0);
            int64_t nextToken = torch::multinomial(probabilities, 1).item<int64_t>();

            tokens64.push_back(nextToken);
            seen.push_back(nextToken);
        }

        std::vector<int> output(tokens64.begin(), tokens64.end());
        return codec.get()->decode(output);
    }

    void save(std::string path, bool includeOptimizer = true) {
        torch::serialize::OutputArchive archive;

        std::vector<torch::Tensor> modelParams = parameters();
        for (size_t i = 0; i < modelParams.size(); i++) {
            torch::Tensor cpu = modelParams[i].detach().to(torch::kCPU);
            archive.write("p" + std::to_string(i), cpu);
        }

        archive.write("dmodel", torch::tensor({(int64_t) dmodel}, torch::kLong));
        archive.write("dff", torch::tensor({(int64_t) dff}, torch::kLong));
        archive.write("heads", torch::tensor({(int64_t) numberOfHeads}, torch::kLong));
        archive.write("blocks", torch::tensor({(int64_t) numberOfBlocks}, torch::kLong));
        archive.write("vocab", torch::tensor({(int64_t) codec.get()->size}, torch::kLong));

        if (includeOptimizer) {
            torch::serialize::OutputArchive optimizerArchive;
            optimizer.get()->save(optimizerArchive);
            archive.write("optimizer", optimizerArchive);
        }

        archive.write("step", torch::tensor({(int64_t) scheduler.get()->step}, torch::kLong));
        archive.write("warmup+total", torch::tensor({(int64_t) scheduler.get()->warmup, (int64_t) scheduler.get()->total}, torch::kLong));
        archive.write("lrs", torch::tensor({scheduler.get()->baseLr, scheduler.get()->minLr}, torch::kDouble));

        archive.save_to(path);
    }

    void load(const std::string path, bool loadOptimizer = true) {
        torch::serialize::InputArchive archive;
        archive.load_from(path);

        std::vector<torch::Tensor> params = parameters();
        for (size_t i = 0; i < params.size(); i++) {
            torch::Tensor t;
            archive.read("p" + std::to_string(i), t);

            t = t.to(device);
            params[i].detach_();
            params[i].copy_(t, true);
            params[i].set_requires_grad(true);
        }

        if (loadOptimizer) {
            try {
                torch::serialize::InputArchive optimizerArchive;
                archive.read("optimizer", optimizerArchive);
                optimizer.get()->load(optimizerArchive);
            } catch (const c10::Error&) {

            }
        }

        try {
            torch::Tensor stepT;
            archive.read("step", stepT);
            scheduler.get()->step = stepT.item<int64_t>();

            torch::Tensor wt;
            archive.read("warmup+total", wt);
            scheduler.get()->warmup = wt[0].item<int64_t>();
            scheduler.get()->total = wt[1].item<int64_t>();

            torch::Tensor lrs;
            archive.read("lrs", lrs);
            scheduler.get()->baseLr = lrs[0].item<double>();
            scheduler.get()->minLr = lrs[1].item<double>();
        } catch (const c10::Error&) {

        }
    } 
};

int main() {
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);

    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    Transformer t(D_MODEL, DFF, N_HEADS, N_BLOCKS);

    try {
        t.load("model.pt", true);
        std::cout << "Save data found. Not training." << '\n';
    } catch (const c10::Error&) {
        std::cout << "Save data NOT found. Starting training." << '\n';
        t.train(EPOCHS, VALIDATION_STEPS, SAVE_STEPS, GRAD_ACCUM);
    }

    std::cout << t.generate("Animal Farm is not the") << '\n';
}
