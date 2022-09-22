import argparse
import os
import re
import numpy as np
from utils import evaluation_tools

# Get input arguments
parser = argparse.ArgumentParser(description=
                                    'Calculate evaluation metrics (PSNR/SSIM/OCR) \
                                        for the deblurred images.'
                                )

parser.add_argument('--deblurred_path', type=str,
                        help='Path with the results of the deblurring algorithm.'
                    )
parser.add_argument('--text_path', type=str, default=None, required=False,
                        help='Path with text files containing the ground truth transcription\
                            of the images (must have the same name of the corresponding deblurred image).'
                    )
parser.add_argument('--groundTruth_path', type=str, default=None, required=False,
                        help='Path with the ground truth images (must have the same name of the\
                        corresponding deblurred image, but can be of different file extension).'
                    )   
parser.add_argument('--calculate_OCR', action='store_true',
                        help='Calculate the OCR metric for the images. Require path to \
                            text files (--text_path).'
                    )
parser.add_argument('--calculate_PSNR', action='store_true',
                        help='Calculate the PSNR metric for the images. Require path to \
                            ground truth images (--groundTruth_path).'
                    )
parser.add_argument('--calculate_SSIM', action='store_true',
                        help='Calculate the SSIM metric for the images. Require path to \
                            ground truth images (--groundTruth_path).'
                    )
parser.add_argument('--omit_progress', action='store_false',
                        help='Omit the print of the execution progress\
                            (current image number/total image number).'
                    )
parser.add_argument('--save_name', type=str, default=None, required=False,
                        help='Name (including desired path) to save the dictionary with the metrics results.'
                    )                                             
                                     
args = parser.parse_args()

# Get image names
r = re.compile(".+\\.")
deblurred_names = sorted(os.listdir(args.deblurred_path))
deblurred_names = list(filter(r.match, deblurred_names))
print(f"{len(deblurred_names)} images were found.")

results = {}

assert(args.calculate_OCR or args.calculate_PSNR or args.calculate_SSIM), "No metric was specified. Use the options\
    --calculate_OCR, --calculate_PSNR and/or --calculate_SSIM to specify the metrics to calculate."

if(args.calculate_PSNR or args.calculate_SSIM):
    assert(args.groundTruth_path is not None), "Calculating the PSNR/SSIM score requires\
                                                 the groundTruth_path argument."

    groundTruth_names = sorted(os.listdir(args.groundTruth_path))
    groundTruth_names = list(filter(r.match, groundTruth_names))

    assert (len(deblurred_names) == len(groundTruth_names)), "Ground truth and deblurred folders have\
                                                            a different number of images."
    if(args.calculate_PSNR):
        results['psnr'] = []

    if(args.calculate_SSIM):
        results['ssim'] = []

if(args.calculate_OCR):
    assert(args.text_path is not None), "Calculating the OCR score requires the text_path argument."

    text_names = sorted(os.listdir(args.text_path))
    text_names = list(filter(r.match, text_names))
    assert (len(deblurred_names) == len(text_names)), "Text and deblurred folders have\
                                                        a different number of files."
    results['ocr'] = []

for i_img in range(len(deblurred_names)):
    deblurredFile = os.path.join(args.deblurred_path, deblurred_names[i_img])

    if(args.calculate_OCR):
        textFile = os.path.join(args.text_path, text_names[i_img])
        OCR_score = evaluation_tools.evaluateImage_OCR(deblurredFile, textFile)
        results['ocr'].append(OCR_score)
    
    if(args.calculate_PSNR or args.calculate_SSIM):
        referenceFIle = os.path.join(args.groundTruth_path, groundTruth_names[i_img])

        PSNR_score, SSIM_score = evaluation_tools.evaluateImage_Quality(deblurredFile, referenceFIle,
                                                                        args.calculate_PSNR, args.calculate_SSIM)
        if(args.calculate_PSNR):
            results['psnr'].append(PSNR_score)
        
        if(args.calculate_SSIM):
            results['ssim'].append(SSIM_score)
    
    if(not args.omit_progress):
        print(f"{i_img}/{len(deblurred_names)}")

calculated_metrics = list(results.keys())

for metric in calculated_metrics:
    results[metric+'_mean'] = np.mean(results[metric])
    results[metric+'_std'] = np.std(results[metric])
    print(f"{metric.upper()} result: {results[metric+'_mean']} +- {results[metric+'_std']}")

if(args.save_name is not None):
    np.save(args.save_name, results, allow_pickle=True)