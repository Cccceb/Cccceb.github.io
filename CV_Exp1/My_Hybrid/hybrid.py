# encoding: utf-8


from PIL import Image
import cv2
import numpy as np
'''
def cross_correlation_2d ( img , kernel ) :
    height,width = img.shape
    h,w = kernel.shape
    new_h = height-h+1
    new_w = width-w+1
    new_img = np.zeros((new_h,new_w),dtype = np.float)
    for i in range(new_w):
        for j in range(new_h):
            new_img[i,j]  = np.sum(img[i:i+h,j:j+w]*kernel)
    new_img = new_img.clip(0,255)
    new_img = np.rint(new_img).astype('uint8')
    return new_img
'''
def cross_correlation_2d ( img , kernel ) :
    img_array = np.array ( img )  # 把图像转换为数字
    print("img : ", img_array.shape)
    #print(img_array)
    print("kernel : ", kernel.shape)
    #print(kernel)

    r = img_array.shape [ 0 ]
    c = img_array.shape [ 1 ]


    r2 = kernel.shape [ 0 ]
    c2 = kernel.shape [ 1 ]

    if img_array.ndim==3:
        h = img_array.shape [ 2 ]  # 图像的高度
        con = np.zeros ( (r , c , h) )
    for i in range ( 3 ) :  # 对矩阵进行一个互相关运算
        tmp=np.zeros((r,c))
        new_img_array = np.zeros ( (r + r2//2*2 , c + c2//2*2) )
        if img_array.ndim==3:
            for j in range ( 0,r ) :
                for k in range ( 0,c ) :
                    new_img_array [ r2 // 2 + j ] [ c2 // 2 + k ] = img_array [ j ] [ k ] [ i ]
        else:
            for j in range ( 0,r ) :
                for k in range (0, c ) :
                    new_img_array [ r2 // 2 + j ] [ c2 // 2 + k ] = img_array [ j ] [ k ]
        #print("new_img",new_img_array.shape)
        #print(new_img_array)
        for j in range(r2//2,r+r2//2):
            for k in range(c2//2,c+c2//2):
                tmp[j-r2//2][k-c2//2] = (new_img_array[j-r2//2:j+(r2+1)//2,k-c2//2:k+(c2+1)//2]*kernel).sum()
                #print("mul ",j," ",k," ",j-r2//2," ",j+(r2+1)//2," ",k-c2//2," ",k+(c2+1)//2)
        if img_array.ndim == 2 :
            print ( "con : " , tmp.shape,"\n")
            #print(tmp)
            return tmp
        con[:,:,i]= tmp
    print ( "con : " , con.shape ,"\n")
    #print(con)
    return con
    # TODO-BLOCK-BEGIN
    raise Exception ( "TODO in hybrid.py not implemented" )
    # TODO-BLOCK-END


def convolve_2d ( img , kernel ) :
    kernel2 = np.fliplr ( np.flipud ( kernel )  )  # 将图片进行2次逆时针90度翻转
    return cross_correlation_2d ( img , kernel2 )
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    raise Exception ( "TODO in hybrid.py not implemented" )
    # TODO-BLOCK-END


def gaussian_blur_kernel_2d ( sigma , height , width ) :
    gaussian_kernel=np.zeros((height,width),dtype='double')
    center_row=height//2
    center_column=width//2
    s=2*(sigma**2)
    sum_val = 0
    for i in range(height):
        for j in range(width):
            x=i-center_row
            y=j-center_column
            gaussian_kernel[i][j]=(1.0/(np.pi*s))*np.exp(-float(x**2+y**2)/s)
            sum_val += gaussian_kernel[i][j]
            #gaussian_kernel[i][j]=(1.0/(np.pi*s))*np.exp(-(i**2+j**2)/s)
    return gaussian_kernel/sum_val # 返回高斯核
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    raise Exception ( "TODO in hybrid.py not implemented" )
    # TODO-BLOCK-END


def low_pass ( img , sigma , size ) :
    height = width = size
    res = gaussian_blur_kernel_2d ( sigma , height , width )  # res为一个高斯核
    #    print_gaussian(res,height,width)#把核函数输出来看看
    return convolve_2d ( img , res )  # 进行卷积
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    raise Exception ( "TODO in hybrid.py not implemented" )
    # TODO-BLOCK-END


def high_pass ( img , sigma , size ) :
    height = width = size
    Image = np.array ( img )
    # print("img : ",Image.shape)
    return (img - low_pass ( Image , sigma , size ))  # 做一个减法得到高通图像
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    raise Exception ( "TODO in hybrid.py not implemented" )
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)


    cv2.imwrite('left.jpg',img1*255)
    cv2.imwrite('right.jpg',img2*255)
    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    if(img1.size < img2.size):
        img2=cv2.resize(img2,(img1.shape[1],img1.shape[0]),interpolation=cv2.INTER_CUBIC)
    elif img1.size>img2.size:
        img1 = cv2.resize(img2,(img2.shape[1],img2.shape[0]),interpolation=cv2.INTER_CUBIC)

    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip ( 0 , 255 ).astype ( np.uint8 )

if __name__ == '__main__' :
    img1 = cv2.imread('./resources/1.jpg')
    img2 = cv2.imread('./resources/2.jpg')
    img1 = np.array(img1)
    img2 = np.array(img2)

    ratio = 0.65
    Img = create_hybrid_image(img1,img2,7.1,8,"low",6.1,8,"high",ratio)
    cv2.imwrite('hybrid.jpg', Img)

