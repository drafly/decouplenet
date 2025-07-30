# DecoupleNet: Domain-Specific Task Decoupling Network for RAW Low-Light Enhancement
This is the official code for the Decoupled Image-domain Multi-Stage Network, which is used for low-light enhancement

## Data Preparation

> To be the same as the [DNF](https://github.com/Srameo/DNF), please follow the instruction in the repository. We only show some basic commands here, and all the `txt files` for training and testing can be found in [Google Drive](https://drive.google.com/drive/folders/1DIuBcbq0wjbzmmSp0XSp7vrnW-jiKLFD?usp=drive_link).

<table>
<thead>
  <tr>
    <th> Dataset </th>
    <th> :link: Source </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td> SID Sony </td>
    <th> <a href='https://cchen156.github.io/SID.html'>Learning to see in the dark</a> (<a href='https://drive.google.com/file/d/1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx/view'>dataset only</a>) </th>
  </tr>
  <tr>
    <td> SID Fuji </td>
    <th> <a href='https://cchen156.github.io/SID.html'>Learning to see in the dark</a> (<a href='https://drive.google.com/file/d/1C7GeZ3Y23k1B8reRL79SqnZbRBc4uizH/view'>dataset only</a>) </th>
       </tr>
  <tr>
    <td> MCR </td>
    <th> <a href='https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark'>Abandoning the Bayer-Filter to See in the Dark</a> (<a href='https://drive.google.com/file/d/1Q3NYGyByNnEKt_mREzD2qw9L2TuxCV_r/view'>dataset only</a>) </th>

## Pretrained Models
You can download the pretrained models at [google drive](https://drive.google.com/drive/folders/1BsRThTCWwZQVAvL-5mDQ40mE8Xax9wFP?usp=drive_link), then put them in the following path
| Trained on | 🔗 Path                           |
| ---------- | -------------------------------- |
| SID Sony   | ./pretrained/sony.pth |
| SID Fuji   | ./pretrained/fuji.pth |
| MCR        | ./pretrained/mcr.pth  |


## Evaluation

```python
python test_sony.py
```
```python
python test_fuji.py
```
```python
python test_mcr.py
```
## Training 

Training from scratch!

```python
python train.py -cfg ./configs/sony.yaml
```





## Acknowledgement



This repository borrows extremely heavily from [DNF](https://github.com/Srameo/DNF)，[BasicSR](https://github.com/XPixelGroup/BasicSR) and [Learning-to-See-in-the-Dark](https://github.com/cchen156/Learning-to-See-in-the-Dark).
