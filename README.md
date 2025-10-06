# tiny-transformer
A transformer made in C++ with Libtorch (C++ version of PyTorch), trained on a dataset of 3,000 words. The dataset is an extract from the book 'Animal Farm' by George Orwell. This was based off of the "Attention Is All You Need" research paper, and nanoGPT implemented by karpathy.

With an Epoch of 250, this model can generate partial sentences.
Example
```
Input: Animal Farm is not the
```
```
Animal Farm is not the namal muster evestion a comfort enemies. And amon animals of that in the firmenise of
England in the had began to slavery. Can you not understand the life of Man is that the
animals he sett of on.
```

In the future, I would like to implement BPE (Byte Pair Encoding) to generate a more sensible output.

**Requirements: Make sure you have a compiled version of torchlib in the project directory before running.**
