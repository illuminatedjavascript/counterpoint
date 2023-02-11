
# counterpoint

A masked sequence model for harmonisation in the style of J.S. Bach. Inspired by other work (1, 2), this model uses a transformer encoder to harmonize incomplete piano-rolls in the style of Bach. See (3) for more information about the dataset.

### Installation

I recommend you create your own environment if you want to install. Entering the following commands should install the repo correctly on Windows/Linux machines.

```
git clone https://github.com/loua19/counterpoint
cd counterpoint
conda create --name counterpoint python
conda activate counterpoint
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install MIDIUtil
pip install progress
```

### Usage

If you want to generate your own samples, simply type ```python run.py sample```. By modifying models/sample.py you can enter your own prompts for re-harmonisation. Simply follow the process outlined in the ```Sampler``` class.

### References

1. Huang, C.-Z. A., Cooijmans, T., Roberts, A., Courville, A., & Eck, D. (2019). Counterpoint by Convolution.

2. Huang, C.-Z. A., Vaswani, A., Uszkoreit, J., Shazeer, N., Simon, I., Hawthorne, C., Dai, A. M., Hoffman, M. D., Dinculescu, M., & Eck, D. (2018). Music Transformer.

3. https://github.com/czhuang/JSB-Chorales-dataset