# Third-party notices

## DeepeR ResUNet Raman denoiser

RamanPhaseID includes an inference implementation of the 1D ResUNet architecture
and can download the pretrained `ResUNet.pt` checkpoint from
[conor-horgan/DeepeR](https://github.com/conor-horgan/DeepeR). The source URL is
pinned to commit `87da149b2cdc8b4d98af60f6211f3b35d3c21493`, and the downloaded
checkpoint is accepted only when its SHA-256 digest equals
`23d11061fce98656f32f8d604d2e58973853a3f79ce69e9f08dac4d8ef9747b2`.

Copyright (c) 2020 conor-horgan

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Scientific reference:

Conor C. Horgan et al., “High-throughput molecular imaging via deep learning
enabled Raman spectroscopy,” *Analytical Chemistry* 93 (2021), 15850–15860,
[doi:10.1021/acs.analchem.1c02178](https://doi.org/10.1021/acs.analchem.1c02178).
