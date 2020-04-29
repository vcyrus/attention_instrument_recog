# An Attention Mechanism for Musical Instrument Recognition
- A reimplementation of the paper ['An Attention Mechansim for Musical Instrument Recognition'](https://arxiv.org/abs/1907.04294) by S Gururani, M Sharma, A Lerch (ISMIR 2019)

## TODO
- [ ] implement partial BCE
- [ ] evaluation

## Installation
- Download the [OpenMic dataset](https://github.com/cosmir/openmic-2018)
- `git clone`
- `cd attention_instrument_recognition/`
- `pip install requirements.txt`

## Running
- `python -m src.scripts.train <path_to_openmic>/openmic-2018/`
- command-line arguments can be found in `src.utils.parse_args`

