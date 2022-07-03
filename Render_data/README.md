## Generating synthetic data for RTL

### Prepare the data for blender

Download the data, which can be found at [here](https://1drv.ms/u/s!AiwRMXEmaB9wiTr1qrZ7ZDSS6u5-?e=F0jYco).  

```
mv /path/for/data  /Render_data
```

### Configure

Download blender-2.79a, and revise the `cfg.BLENDER_PATH` in the `config.py`.

revise the `cfg.ROOT_DIR`  in the `config.py`.

### Run

**All the Commands must be carried out in the ROOT directory.**

images from blender rendering

```
python run.py
```

### Examples

Blender rendering

![blender](README.assets/rtl-16568636709302.png)

