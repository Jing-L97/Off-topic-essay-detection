# Off-topic-essay-detection
This project proposes an innovative approach for off topic essay detection by converting the similarity between the on topic sample essay and the target essay into similarity grids on word level, and then classifying the extent of topic relevance with residual neural networks (ResNet). This novel method achieved F1 scores of 92% and 95.8% based on different semantic features, which was substantially higher than the F1 score of a baseline system Random Forest Classifier (83.7%). 
For the paper, please refer to this link https://sites.google.com/view/jliucog/research?authuser=0. 

## License
- Essay data were collected from HSK corpus (the Chinese language Proficiency Test), all of which were written by HSK exam takers (foreign language learners of Mandarin Chinese) from 1992 to 2005. They cover wide range of required topics including family, communication, education, people, music, health, and technology with average length of 118 words. Each text is annotated with the essay grades, required topics, and grammatical mistakes by expert human raters. 
For the file name, 0 stands for off-topic essays and 1 stands for on-topic essays
- This image similarity model is developed based on Keras deep learning model from https://github.com/fchollet/deep-learning-models

## Citations
```
@inproceedings{
  author = {Jing Liu},
  title = {Automatic detection of off-topic essays with ResNet CNN},
  year = {2021}
}
```
