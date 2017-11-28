##Texturing on source image

Given a source.png and a texture.jpg/png, it would generate a new image which has source's content and texture's texture.

###Execute commend

`python texturedraw.py sourcefile texturefile targetfile`

The result will be saved according to targetfile.

###Examples

source image:

![](https://github.com/ruozhichen/Texturing/blob/master/input/bg2.png)

texture image:

![](https://github.com/ruozhichen/Texturing/blob/master/texture/t3.jpg)

output image:

![](https://github.com/ruozhichen/Texturing/blob/master/output/bg2_t3_lab.png)

##Color Space Conversion

LAB.py and HSL.py are for color space conversion

LAB.py includes conversion between rgb and lab

HSL.py includes conversion between rgb and hsl

More details can be seen in [ruozhichen\\rgb2Lab-rgb2hsl](https://github.com/ruozhichen)

