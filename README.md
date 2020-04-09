# sepal : Identifying Transcription Profiles with Spatial Patterns by Diffusion-based Modeling

This repo contains:
* the sepal python package, and installation files
* tutorials and examples of usage (CLI and API)
* data used in the publication
* results presented in the publication


## Download and Install
`sepal` requires `python3`, preferably a version later than or equal to 3.5. To
download and install, open the terminal and change to a directory where you want
`sepal` to be downloaded to and do:

```sh

git clone https://github.com/almaan/sepal.git
cd sepal
chmod +x setup.py
./setup.py install

```
Depending on your user privileges, you may have to add `--user`  as an argument to `setup.py`.
Running the setup will give you the minimal required install to compute the diffusion times. However,
if you want to be able to use the analysis modules, you also need to install the recommended packages.
To do this, simply (in the same directory) run:

```shape
pip3 install "sepal[full]" -e
```
again, you may have to provide the `--user` flag, depending on user privileges.

This should install both a command line interface (CLI) and a standard package.
To test and see whether the installation was successful you could try executing the command:

```sh
sepal -h

```
Which should print the help message associated with sepal. If everyhing wored out for you so far,
you may proceed to the example section to see `sepal` in action!

## Examples

### CLI 

The recommended usage of sepal is by 
the command line interface. Both the simulations
in order to compute the diffusion times as well
as subsequent analysis or inspection of the results
can easily be performed by typing `sepal` followed by 
either `run` or `analyze`. The `analyze` module have different
options, to visualize the results (`inspect`),
sort the profiles into pattern families (`family`) or subject 
the identified families to functional enrichment analysis. For a 
complete list of commands available, do  `sepal module -h`, where module
is one of `run` and `analyze`. Below, we illustrate
how sepal may be used to find transcription profiles with spatial patterns.

We will create a folder for our results, which will also figure
as our working directory. From the main directory of the repo, do:
```sh
cd res
mkdir example
cd example
```

The MOB sample will be used to exemplify our analysis. We begin
with calculating diffusion times for each transcription profile:

```sh
sepal run -c ../../data/real/mob.tsv.gz -mo 10 -mc 5 -o . -ar 1
```
Below is an example (with an additional display of the help command)
of how this might look

![CLI run example][run_ex]

Having computed the diffusion times, we want to inspect the result, like
in the study, we will look at the top 20 profiles. We can easily generate
images from our result by running the command:
```sh
 sepal analyze -c ../../data/real/mob.tsv.gz \
-r 20200409173043610345-top-diffusion-times.tsv \
-ar 1k -o . inspect -ng 20 -nc 5
```
Which would look something in the line of this:

![CLI analyze example][anl_ex]

The output wil be the following image:

![Analysis output][viz_ex]

Then, to sort the 100 top ranked genes into
a set of pattern families, where 90% of the variance in our patterns
should be explained by the eigenpatterns, running

```sh
sepal analyze -c ../../data/real/mob.tsv.gz \
-r 20200409173043610345-top-diffusion-times.tsv \
-ar 1k -o . family -ng 100 -nbg 100 -eps 0.90 --plot -nc 3
```

From this, we obtain the following three 
representative motifs for each family:

![Representative motifs][mob-motif]


[anl_ex]: https://github.com/almaan/sepal/blob/master/img/analyze-ex.gif?raw=true
[run_ex]: https://github.com/almaan/sepal/blob/master/img/run-example.gif?raw=true
[viz_ex]: https://github.com/almaan/sepal/blob/master/img/mob-ex.png?raw=true
[mob-motif]: https://github.com/almaan/sepal/blob/master/img/mob-motif.png?raw=true

We may subject our families to enrichment analysis, by running:

```sh
sepal analyze -c ../../data/real/mob.tsv.gz \
      -r 20200409173043610345-top-diffusion-times.tsv \
   -ar 1k -o . fea -fl mob.tsv-family-index.tsv -or "mmusculus"
```

where we for example see that Family 0 is enriched for several processes related to neuronal function, generation and regulation:

|    |   family | native     | name                                                               |     p\_value | source   |   intersection_size |
|---:|---------:|:-----------|:-------------------------------------------------------------------|------------:|:---------|--------------------:|
|  1 |        0 | GO:0007399 | nervous system development                                         | 5.59915e-06 | GO:BP    |                  30 |
|  2 |        0 | GO:0050773 | regulation of dendrite development                                 | 0.000112462 | GO:BP    |                   9 |
|  3 |        0 | GO:0048666 | neuron development                                                 | 0.000297393 | GO:BP    |                  19 |
|  4 |        0 | GO:0016358 | dendrite development                                               | 0.000425748 | GO:BP    |                  10 |
|  5 |        0 | GO:0048814 | regulation of dendrite morphogenesis                               | 0.000765306 | GO:BP    |                   7 |
|  6 |        0 | GO:0048813 | dendrite morphogenesis                                             | 0.00100365  | GO:BP    |                   8 |
|  7 |        0 | GO:0010975 | regulation of neuron projection development                        | 0.00131963  | GO:BP    |                  13 |
|  8 |        0 | GO:0031175 | neuron projection development                                      | 0.0014279   | GO:BP    |                  17 |
|  9 |        0 | GO:0048167 | regulation of synaptic plasticity                                  | 0.00375046  | GO:BP    |                   8 |
| 10 |        0 | GO:0030182 | neuron differentiation                                             | 0.00615277  | GO:BP    |                  19 |
| 11 |        0 | GO:0120036 | plasma membrane bounded cell projection organization               | 0.0106412   | GO:BP    |                  19 |
| 12 |        0 | GO:0050804 | modulation of chemical synaptic transmission                       | 0.0110822   | GO:BP    |                  11 |
| 13 |        0 | GO:0099177 | regulation of trans-synaptic signaling                             | 0.011293    | GO:BP    |                  11 |
| 14 |        0 | GO:0030030 | cell projection organization                                       | 0.015298    | GO:BP    |                  19 |
| 15 |        0 | GO:0010769 | regulation of cell morphogenesis involved in differentiation       | 0.0159329   | GO:BP    |                   9 |
| 16 |        0 | GO:0045664 | regulation of neuron differentiation                               | 0.0185319   | GO:BP    |                  13 |
| 17 |        0 | GO:0099004 | calmodulin dependent kinase signaling pathway                      | 0.0228571   | GO:BP    |                   3 |
| 18 |        0 | GO:0120035 | regulation of plasma membrane bounded cell projection organization | 0.0228766   | GO:BP    |                  13 |
| 19 |        0 | GO:0031344 | regulation of cell projection organization                         | 0.0262513   | GO:BP    |                  13 |
| 20 |        0 | GO:0051960 | regulation of nervous system development                           | 0.0291213   | GO:BP    |                  15 |
| 21 |        0 | GO:0048699 | generation of neurons                                              | 0.0307889   | GO:BP    |                  19 |
| 22 |        0 | GO:0050808 | synapse organization                                               | 0.0321188   | GO:BP    |                  10 |
| 23 |        0 | GO:0050767 | regulation of neurogenesis                                         | 0.0366861   | GO:BP    |                  14 |
| 24 |        0 | GO:0098916 | anterograde trans-synaptic signaling                               | 0.0372187   | GO:BP    |                  12 |
| 25 |        0 | GO:0007268 | chemical synaptic transmission                                     | 0.0372187   | GO:BP    |                  12 |
| 26 |        0 | GO:0099537 | trans-synaptic signaling                                           | 0.0417471   | GO:BP    |                  12 |


Of course, this analysis is by no means extensive. But rather an quick example to show how one operates the CLI for  `sepal` .

### As imported package

While `sepal` has been designed as a standalone tool, we've also constructed it
to be functional as a standard python package from which functions may be
imported and used in an integrated workflow. To show how this may be done, we
provide 2 examples contained in jupyter notebooks:

* Melanoma analysis  : [LINK](https://github.com/almaan/sepal/blob/master/examples/melanoma.ipynb)
* Breast Cancer analysis : [LINK]()

## Data
All the data we used is public, and can be found accessed at the following links:

* MOB : [LINK][1], use Rep11
* Mouse Brain : [LINK][2] 
* Lymph Node :[LINK][3]
* Melanoma : [LINK][4],use ST\_mel1\_rep1
* Cerebellum :[LINK][5], use Cerebellum\_Puck\_180819\_11

However, you may also find these sets, processed and ready to use
in the `data` directory.



[1]: https://www.spatialresearch.org/resources-published-datasets/doi-10-1126science-aaf2403/
[2]: https://support.10xgenomics.com/spatial-gene-expression/datasets/1.0.0/V1_Adult_Mouse_Brain
[3]: https://support.10xgenomics.com/spatial-gene-expression/datasets/1.0.0/V1_Human_Lymph_Node
[4]: https://www.spatialresearch.org/resources-published-datasets/doi-10-1158-0008-5472-can-18-0747
[5]: https://singlecell.broadinstitute.org/single_cell/data/public/SCP354/slide-seq-study

## Results
