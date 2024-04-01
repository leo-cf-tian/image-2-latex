In this project, we aim to be able to design an OCR handwritten mathematical equations into a LaTeX format. There are already existing models able to reliably convert images of LaTeX equations back into LaTeX, but it seems like a converting handwritten equations reliably is the harder task.

The idea then, is to train a Vision Transformers model on LaTeX images, then fine-tune it for handwritten equations.

# Data & Preprocessing

## Sources

As there is much less handwritten data than LaTeX data, we opt to pre-train the model using LaTeX equations.

| Data                                                                                                            | Dataset Size | Notes                                                                                                                                               | Task        |
| --------------------------------------------------------------------------------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| [im2latex-100k](https://huggingface.co/datasets/yuntian-deng/im2latex-100k)                                     | ~60k         | HuggingFace Dataset<br>LaTeX images with LaTeX ground truth                                                                                         | Pretraining |
| [im2latex-230k](https://www.kaggle.com/datasets/gregoryeritsyan/im2latex-230k)                                  | ~230k        | Kaggle Dataset<br>LaTeX images (includes matrix) symbols with LaTeX ground truth<br><br>**Not used due to low data quality and similarity to 100k** | Unused      |
| Wikipedia Dataset                                                                                               | 100k         | Scraped this myself                                                                                                                                 | Pretraining |
| [Handwritten Math Equations](https://www.kaggle.com/datasets/rtatman/handwritten-mathematical-expressions)      | ~1.1k<br>    | Kaggle Dataset<br>Data is in InkML, which is composed of pen strokes and a MathML ground truth                                                      | Downstream  |
| [Aida Calculus Math Handwriting Recognition Dataset](https://www.kaggle.com/datasets/aidapearson/ocr-data/data) | ~100k        | Kaggle Dataset<br>Handwritten expression images (containing calculus only) with character bounding boxes and LaTeX ground truth                     | Downstream  |
## Wikipedia Scraper

Wikipedia remains a good source of data due to
- Diversity: Data contains much more diverse representations
- Representation: Wikipedia latex is much more naturally written
- Consistency: Wikipedia has consistent syntax
- Verification: Wikipedia latex is almost always rendered correctly, as it is checked by humans

Thus a system to scrape Wikipedia was developed, it can be found at [`latex-scraper.ipynb`](./latex-scraper.ipynb)

# Vision Transformer Pre-Training

### Tokenization

We use a Byte-Level BPE for encoding. Word-level tokenizers were tested but yielded significantly worse performance. The reason is not yet known, but it is possibly because of the whitespace pre-processor, which leads to the model incorrectly handling whitespace characters during training.

### Pretraining Custom Decoder

For the decoder, we train a custom RoBERTa model with Masked Language Modelling, using the training data formulas as the corpus.

The data was reorganized after MLM training, further investigation will be needed to determine whether this caused Data Leakage.

### Vision Encoder-Decoder Model

The both the image processor and the encoding layers of the transformer were extracted from [Google's ViT Implementation](https://huggingface.co/google/vit-base-patch16-224-in21k).

The model was then trained partially on the original data, and partially on augmented data with the following augmentations (using PyTorch transforms).

- Random sharpness adjustments
- Random rotation
- Random perspective
- Elastic deformation
- Color jitter
- Random color inversion

Which simulates future conditions wherein paper color is not white, and where text may be warped.

**Resources**
- [Image Captioning Using Hugging Face Vision Encoder Decoder](https://medium.com/@kalpeshmulye/image-captioning-using-hugging-face-vision-encoder-decoder-step-2-step-guide-part-1-495ecb05f0d5)
- [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train)
- [Create a Tokenizer and Train a Huggingface RoBERTa Model from Scratch](https://medium.com/analytics-vidhya/create-a-tokenizer-and-train-a-huggingface-roberta-model-from-scratch-f3ed1138180c)
- [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/pdf/2109.10282.pdf)
## Pre-Training Result Analysis

Testing the model post-pre-training reveals:

**The model performance is acceptable**
- The model generally does not make breaking mistakes, and generally recognizes examples correctly.

**The model generalizes reasonably well on augmented images**
- Performance on images with noticeable but not significant warping being equal or above to performance on normal images.

**The model cannot understand images that have significant padding** (FIXED)
- [x] Add randomized padding to preprocessing
- If model needs to extract latex from larger images, an R-CNN#Faster R-CNN|RPN may be required.

**The model generalizes poorly to images with long equations**
- The model performs badly 400px in length, but can correctly identify segments
	- Processor resolution limitations, possible sliding windows solution
- Eliminate training image quality suspicions

**The model cannot handle short sequences**
- Definitely an overfitting issue

**The model cannot handle stacked fractions**
- Model requires better spatial representation

**The model perceives unnecessary spaces in large images**
- More data augmentation required

**The model does not correctly identify strings of text
- More data required

**The model has a hard time recognizing non-greyscale text**
- More data augmentation required

# Things to Test / Implement

- [ ] **Attention Rollout**
	- I want to see what the model sees

- [ ] **Resolving Limited Resolution Issue**
	- [Conditional Positional Encodings for Vision Transformers](https://arxiv.org/pdf/2102.10882.pdf)
	- ~~[CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/pdf/2103.15808.pdf)~~
		- CvT Convolutional Projection layers lower number of tokens and raise token feature size, making it unfit for typical encoder-decoder architectures

- [ ] **Convolutional Network Backbone**
	- Just seems like a common implementation worth a try, could be good for feature extraction

- [ ] **Correction Model**
	- [A Transformer-based Math Language Model for Handwritten Math Expression Recognition](https://arxiv.org/ftp/arxiv/papers/2108/2108.05002.pdf)
	- Some type of attention-based model where the model makes error proposals, and checks the original image to resolve the issues.

- [ ] **Two-stage model: character detection model for spatial representations and preliminary classification**
	- [Offline handwritten mathematical expression recognition with graph encoder and transformer decoder](https://pdf.sciencedirectassets.com/272206/1-s2.0-S0031320323X00120/1-s2.0-S003132032300852X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEC8aCXVzLWVhc3QtMSJIMEYCIQCBGHUsNWT5s542owQ%2B6zfn7mNHODiDSS%2FywybmlXl2VwIhAOie6Vj%2BrPRL1dPKFiE6i3i%2BCj6diPmAg9r63kz2iW1UKrMFCCcQBRoMMDU5MDAzNTQ2ODY1IgyFQ44qVghVwpN031kqkAVgfDR3f0F5PjMsNnbfQlClE11%2F5OPAgfAUhXH9zQs9stFlDqjlWd2teCQ3TfuYhYrIRybdXGW5gF433ILAVyuRn24qa4FSVWjF6XL%2FNLasGw1sUZiq5RyiXeufp8XxEuxdTsNHtfJCcdbSFGrtC%2Fh7diAUmhpcVViVEOaAir1awnE57AVZ9aldq54sW7M6%2FhFTo3jiZNedWfAG9f%2BfHvdNfc69D%2FRAeR6ls3TZoOUsFTvm2LqZr9k92mxOMuLq9SgggnMxbfDBxTfe9Okr5cwEMJfvWju%2F%2F2nKxvT%2FMImc3R%2BxUjElgEevuJ5f7IFKf8BakCIyUb42etlDHiKcvYUn%2BELme%2FRPuMqdi4OgoDU%2BElxvASVnWyRF8G%2FdeA%2BWa7TrrOiVFuE3rgWZ%2BYvUAdl27BT5l%2BXX0%2Bi9kFzYh8mYxx1QE4fy5lZisyzHZvHvFJ9vANx%2Bq%2Br7wrlmwq4s%2F84lagdZ9lPr8XmRwXhCUYl%2BkJaqKFXrNEmFbMY1BIh6iKfkNa9n3pHdW1Si1PteA0%2BnDjSTVlQQ269NhyaDtoRc5ZnF2ilbl4Ab7qxfi%2BOXSEisT%2FVV%2FMZU2rXIIYd5I1d9vx4JfY0jlYe1UJ77%2F2R9knQmVatN2txNZ7tcTa3rJCtoRIrPigMuGTnirCC5u7PGPYI%2FpSk3x4MUmZXVZq97wwPqWYfZSn7ibuBbgRQg3uH8PoAis0Lar2y5Xu6u87yRTkm43T3%2FvVZrvQT5WrAYQlkhXkpjHfH%2F4q1U%2FjDkXjHTkcYQUL7Znvm9NKGMBmurPAATZNieYu0fbSm%2BJuBC%2F5HVeRsncjqGjD1b6U5m3d4x0ACPulsWODMd6MyyEavDi4GlvKpDmsQk8YFwOPo0kDDZ5oWvBjqwAXRR2JQS4I3A%2F1b8A8dZ%2BsJRNzq3IErnDo%2FyiXyMqnCbCQg8Fu24vqSe9t0Vbgy%2F3kR3v2X4Z768t4iA9lD96SYyXGpbIu%2BTzqTE3Pt%2BQNc%2B7gvRe5JJTV1bjwkBeIuRGiHuOpzB9PeZyqvqE%2F20W%2BVxeXuDYsTosbpHtJOxikrSvO301GXnpU4WAAGhiBmT7u2EHuCOzgyLJ%2BsywZB%2FZkn9RZs09g%2BJQak2Q2Q9XwDA&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240301T081634Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY5ZZZ6VHW%2F20240301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=eaca62130ebe479fafb9aa8ad8bfda215fcdacbd5942a359080e52a11eb66696&hash=7b9c6ca3aa36134c1474ecf5cd398c33f791c198704485fe623c01033fe9262e&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S003132032300852X&tid=spdf-a4291622-a449-4d7f-a8ca-8d4b834419d4&sid=61d35c6b744b2143d4291635e3c177ceb026gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=190a5c500750045b0350&rr=85d7b4a7b96a7ab5&cc=cn)
	- [Syntactic data generation for handwritten mathematical expression recognition](https://www.sciencedirect.com/science/article/pii/S0167865521004293/pdfft?md5=a38fc2bb18762eb67a8da79228ac8ca7&pid=1-s2.0-S0167865521004293-main.pdf)


## Possible Leads to Research

**List of papers to go over**
- [Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer](https://arxiv.org/pdf/2105.02412.pdf)
- [Mathematical expression recognition using a new deep neural model](https://pdf.sciencedirectassets.com/271125/1-s2.0-S0893608023X00082/1-s2.0-S0893608023004653/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDAaCXVzLWVhc3QtMSJGMEQCIHa7dVQf%2BMLXV2bb4rRPUqNUUDBk%2FpI%2Bav%2FNk%2BTW3v%2FAAiBYXg1ACuvY%2Fbu3lzNf%2Bi8IAmRqHQuer7qrkDJXoCRClCqzBQgpEAUaDDA1OTAwMzU0Njg2NSIMDGwPBjWrlO%2FwDPYrKpAFvTN0GeHdrY6mubeQxx6uof3HmWbg4k2mfwsL6WNZ%2F8pK1KLkN8rz0y2MeDhalryQTzOW4HsVZ39EzkpB5VJjo%2BWrUmTSlMZXpYoZNobpt4mAe%2F6dpxhzbCEFm5JqWILw0DiyT7PX79QfJUu8tEaHTLVbBSt3TYJQNoc2BWHxLTpJ55P4ANTTzR9IMa63ReTUJTOkaCf7HmkPmvEjSBmb0vBAUFpB3HF3S1rQMDtGa5Gvy%2BayS%2FdFvNzrdMDOGDDAsCTmnsmOYGuiGF4WDc17E%2F08mE8cZrEIWb89YAl3r6V%2FHY1Yjp4VUzPagZgw2NR%2FvEkdwK6s%2FZVN2Y1rBM0MekXPWf2eYxLIpHWzfsQsyjGLk1KJumgIFf6H3ZJ9KmO26z8fvNTe2lsIE63YtPCgpudeQLCuGtyfcmvumQ4%2FIO2uSSL5WX%2Bo0Mkfo5JhmdWR8eCAvLLRdGGBzHAxKxeyTOdpXozlRYHSwwUE10JCmzNwxuEj%2FWkCGN3%2F8eS7lz%2BHhSjHwkDaXF8ZixDNDotQH279K4fwqMOeUTLctjKbxQyJhy4NpjXggvG%2B7JF%2ByNp3pLCiJXzaF9CH8Tze1Bt6tt8BPMji9RJIgU%2FqAIrRVFkk4BS%2BHGWv7MDY9arAM2MnCetnXmio16VdjjaFdcjat3%2BKm%2BFzYgghM9MlUQOUuYguOmWOEIwr%2FmihQnazjG1FEcwJCUxg3Jski4XKNgO%2F4jcWkZHLWQBDQ612glu2pqKmdxFVnmXEgk22uq%2BJXogqZXbzIV4aTpnKNhry21%2B%2FkMiwNdSR%2BoJ0q57fuQC%2FfCp2yg3RTAyeqcisStbb0Zgd6Q5tDAxZGjxJKC4ezK8MCumvUUvhhGvGOJbDtZwaoZ4wo5KGrwY6sgGMnCS1H8NdZp3QduNdLLjnbrFxMkQEv3sPfKCu6IYAcBvhSLciobQ45xNKARMLyFuAIKJttazMGQDZvXXMpaHiAZ3UW7fV4FfBz6SjVAUUUIoXG8Q0iKF%2FSx53J49Dx8d4CdoAjsxtJ6VXGrJAvfEJMr%2BhOr96C%2B5IpZeQXtmRQVXSR103PlqniJPf%2B9Vqoa88Z%2Fz8IrW6wxS%2B7acxyYpd%2FTmwFS2c3rcMI9dg9rlTWBy%2B&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240301T082615Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZLHFJW2I%2F20240301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=2521a0d165d87fe634021f5f96af062bebddd31354b6249eaae3036cbfb9c070&hash=aacbe4a10c63d0ea292d18d163f9f5f2eb21ff70d5aa41e91053d9361323ec25&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0893608023004653&tid=spdf-8404b4b8-d27b-4e09-827f-2ac44c566ae2&sid=61d35c6b744b2143d4291635e3c177ceb026gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=190a5c500750055d0651&rr=85d7c2d67b59fa4a&cc=cn)
- [ICDAR 2023 CROHME: Competition on Recognition of Handwritten Mathematical Expressions](https://hal.science/hal-04264727v1/document)
- [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/pdf/2103.10697.pdf)