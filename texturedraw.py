# -*- coding: utf-8 -*-
"""
@author Sunny.Xia
@Copyright (c) 2017

给定原图、纹理图、结果图地址
args: source(png), texture(jpg or png), target(png)
advise: it would be better if texture's size is similar to source,if possible
"""
import sys
import math
import numpy as np
import scipy.misc
import HSL
import LAB
import time

class Texturing(object):
    """docstring for Texturing"""
    def __init__(self,sourcefile,texturefile):
        #super(Texturing, self).__init__()
        self.source = scipy.misc.imread(sourcefile)   # png
        self.texture = scipy.misc.imread(texturefile)   # jpg or png both ok
        #self.text_scale_adjust()  # 先从texture裁剪出和source相同比例尺寸的区域，再进行缩放

    '''
        由于纹理图的比例不一定和原图相同，导致缩放的话纹理会被拉长或缩短造成失真，所以得先从纹理图中取出合适比例的区域作为纹理参考，之后再进行缩放至原图大小
        不过截取一部分区域，就表明截取的图片比原来的纹理图小，缩小还好，但一般都是放大，这样放大的倍数也就加大，纹理效果虽然是更加明显了，但也能看出明显的放大模糊后果
    '''
    def text_scale_adjust(self):
        scale=self.source.shape[0]*1.0/self.source.shape[1]
        height=int(self.texture.shape[1]*scale)
        if height<=self.texture.shape[0]:
            width=self.texture.shape[1]
            tmp=np.zeros((height,width,3),dtype='int')
            gap=(self.texture.shape[0]-height)/2
            tmp[:,:,:]=self.texture[gap:gap+height,0:width,0:3]  # 取中心区域
        else:
            height=self.texture.shape[0]
            width=int(height/scale)
            tmp=np.zeros((height,width,3),dtype='int')
            gap=(self.texture.shape[1]-width)/2
            tmp[:,:,:]=self.texture[0:height,gap:gap+width,0:3]  # 取中心区域
        self.texture=tmp
    '''
    based Lab color space
    source's L + textures's ab channel + source's alpha channel -> new img with texture
    '''
    def texturingOnSource_Lab(self,targetfile):
        texture_resize=scipy.misc.imresize(self.texture,(self.source.shape[0],self.source.shape[1]))
        texture_copy=texture_resize.reshape(texture_resize.shape[0]*texture_resize.shape[1],texture_resize.shape[2])
        source_copy=self.source.reshape(self.source.shape[0]*self.source.shape[1],self.source.shape[2])
        texture_lab=LAB.rgb2lab_matrix(texture_copy[:,0:3])
        source_lab=LAB.rgb2lab_matrix(source_copy[:,0:3])
        texture_lab[:,0]=source_lab[:,0]
        out=LAB.lab2rgb_matrix(texture_lab)
        #source_copy[:,0:3]=out[:,0:3]  # 修改source_copy的值会影响self.source的值，因此不能对source_copy进行修改
        res=np.zeros((self.source.shape[0]*self.source.shape[1],self.source.shape[2]),dtype='int')
        res[:,0:3]=out[:,0:3]
        res[:,3]=source_copy[:,3]
        target=res.reshape(self.source.shape[0],self.source.shape[1],self.source.shape[2])
        scipy.misc.imsave(targetfile,target)
    '''
    based hsl color space
    source's saturation and lightness + textures's hue channel + source's alpha channel -> new img with texture
    it's effect is worse than Lab method
    '''
    def texturingOnSource_hsl(self,targetfile):
        texture_resize=scipy.misc.imresize(self.texture,(self.source.shape[0],self.source.shape[1]))
        texture_copy=texture_resize.reshape(texture_resize.shape[0]*texture_resize.shape[1],texture_resize.shape[2])
        source_copy=self.source.reshape(self.source.shape[0]*self.source.shape[1],self.source.shape[2])
        texture_hsl=HSL.rgb2hsl_matrix(texture_copy[:,0:3])
        source_hsl=HSL.rgb2hsl_matrix(source_copy[:,0:3])
        texture_hsl[:,1:3]=source_hsl[:,1:3]
        out=HSL.hsl2rgb_matrix(texture_hsl)
        #source_copy[:,0:3]=out[:,0:3]
        res=np.zeros((self.source.shape[0]*self.source.shape[1],self.source.shape[2]),dtype='int')
        res[:,0:3]=out[:,0:3]
        res[:,3]=source_copy[:,3]
        target=res.reshape(self.source.shape[0],self.source.shape[1],self.source.shape[2])
        scipy.misc.imsave(targetfile,target)

def main2(sourcefile,texturefile,targetfile1,targetfile2):
    t=Texturing(sourcefile,texturefile)
    t.texturingOnSource_Lab(targetfile1)
    t.texturingOnSource_hsl(targetfile2)

def main(sourcefile,texturefile,targetfile):
    t=Texturing(sourcefile,texturefile)
    t.texturingOnSource_Lab(targetfile)

if __name__ == "__main__":
    #logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    #start=time.time()
    print sys.argv
    main(sys.argv[1],sys.argv[2],sys.argv[3])
    
    '''
    imgfile="people1"
    textfile="t3"
    start=time.time()
    for i in range(1,10):
        main2("input/%s.png"%imgfile,"texture/t%d.jpg"%i,"output/%s_t%d_lab_2.png"%(imgfile,i),"output/%s_t%d_hsl_2.png"%(imgfile,i))
    end=time.time()
    print end-start
    '''
