# Repositório utilizado para o projeto de pesquisa "Realce de pontos de bifurcação em imagens de vasos sanguineos"

O projeto consiste na comparação de um  metodo de detecção de bifurcação em imagens de vasos sanguineos, aplicados a dois métodos diferentes de segmentação de imagens de retina. O projeto foi realizado como uma Iniciação Científica com apoio da FAPESP. 

## Os métodos implementados se encontram em:

`CALVO, David et al. Automatic detection and characterisation of retinal vessel tree bifurcations and crossovers in eye fundus images. Computer methods and programs in biomedicine, v. 103, n. 1, p. 28-38, 2011.`

`XIANCHENG, Wang et al. Retina blood vessel segmentation using a U-net based Convolutional neural network. Procedia Comput Sci, p. 1-11, 2018.`

## Um breve resumo dos .py presentes neste projeto:

### Run
Um script para executar de uma vez todas as etapas da detecção de bifurcações: Aplicar a detecção, validar resultados e apresentar resultados.

### Stentiford
Método de esqueletização aplicado no método de segmentação proposto por Calvo. Retirado do repositório: https://github.com/ShreyaPrabhu/Thinning-Algorithms e modificado para se adequar as imagens utilizadas.

### Segmentaçao_Calvo
Implementação do método de segmentação proposto por Calvo.

### Calvo
Implementação do método de detecção de bifurcação proposto por Calvo.







