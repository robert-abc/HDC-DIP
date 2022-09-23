import numpy as np
from PIL import Image
import pytesseract
from fuzzywuzzy import fuzz
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import autoencoder_tools

def normalize(img):
    """
    Linear histogram normalization
    source: https://fips.fi/HDC2021_ocrscorecode.zip
    """
    arr = np.array(img, dtype=float)

    arr = (arr - arr.min()) * (255 / arr[:, :50].min())
    arr[arr > 255] = 255
    arr[arr < 0] = 0

    return Image.fromarray(arr.astype('uint8'), 'L')

def evaluateImage_OCR(imageFile, trueTextFile, norm=True):
    """
    OCR evaluation
    source: https://fips.fi/HDC2021_ocrscorecode.zip
    """
    with open(trueTextFile, 'r') as f:
        trueText = f.readlines()

    # remove \n character
    trueText = [text.rstrip() for text in trueText]

    # load image and convert to grayscale
    img = Image.open(imageFile)
    w, h = img.size

    if (norm):
      img = normalize(img)

    # resize image to improve OCR
    img = img.resize((int(w / 2), int(h / 2)))
    w, h = img.size

    # run OCR
    options = r'--oem 1 --psm 6 -c load_system_dawg=false -c load_freq_dawg=false  -c textord_old_xheight=0  -c textord_min_xheight=100 -c ' \
              r'preserve_interword_spaces=0'
    OCRtext = pytesseract.image_to_string(img, config=options)

    # removes form feed character  \f
    OCRtext = OCRtext.replace('\n\f', '').replace('\n\n', '\n')

    # split lines
    OCRtext = OCRtext.split('\n')

    # remove empty lines
    OCRtext = [x.strip() for x in OCRtext if x.strip()]

    # check if OCR extracted 3 lines of text
    #print('File:' + imageFile)
    #print('True text (middle line): %s' % trueText[1])

    if len(OCRtext) != 3:
        #print('ERROR: OCR text does not have 3 lines of text!')
        #print(OCRtext)
        return 0
    else:
        score = fuzz.ratio(trueText[1], OCRtext[1])
        #print('OCR  text (middle line): %s' % OCRtext[1])
        #print('Score: %d' % score)

        return float(score)

def normalize_maxmin(img):
    """
    Normalization applied before calculating PSNR and SSIM metrics
    """
    arr = np.array(img, dtype=float)

    arr = (arr - arr.min()) / (arr.max()-arr.min())*255

    return arr.astype('uint8')

def evaluateImage_Quality(imgFile, refFile, calc_PSNR, calc_SSIM):
    """
    PSNR and SSIM evaluation
    """
    ref = normalize_maxmin(Image.open(refFile))
    img = normalize_maxmin(Image.open(imgFile))
    
    T = autoencoder_tools.get_transform(img,ref)
    ref2 = autoencoder_tools.apply_transform(ref,T)
    ref2 = normalize_maxmin(ref2)
    
    PSNR_score = None
    SSIM_score = None
    
    if(calc_PSNR):
        PSNR_score = psnr(ref2, img, data_range=255)

    if(calc_SSIM):
        SSIM_score = ssim(ref2, img, data_range=255)

    return PSNR_score, SSIM_score
