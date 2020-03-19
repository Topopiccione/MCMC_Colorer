
# MCMC colorer
![banner]
### A parallel algorithm for balanced graph coloring

Based on the article:  
D. Conte, G. Grossi, R. Lanzarotti, J. Lin, A. Petrini  
A parallel MCMC algorithm for the Balanced Graph Coloring problem  
IAPR International workshop on Graph-Based Representation in Pattern Recognition, 19-21 July 2019, Tours, France.

Software implementation by: N. Aspes - G. Grossi - A. Petrini  
PhuseLab / AnacletoLab - Universita' degli studi di Milano - 2019-20

### Table of Contents
<pre>
<a href="#Overview">Overview</a>
<a href="#Requirements">Requirements</a>
<a href="#Downloading-and-compiling">Downloading and compiling</a>
<a href="#Running-MCMC-colorer">Running MCMC colorer</a>
	<a href="#Command-line-options">Command line options</a>
	<a href="#Data-format">Data Format</a>
	<a href="#Random-dataset-generation">Random dataset generation</a>
	<a href="#Examples">Examples</a>
<a href="#License">License</a>
</pre>

---
### Overview
This repository contains a software implementation of the MCMC coloring algorithm presented in the paper:  
D. Conte, G. Grossi,R. Lanzarotti, J. Lin, A. Petrini  
A parallel MCMC algorithm for the Balanced Graph Coloring problem  
IAPR International workshop on Graph-Based Representation in Pattern Recognition, Tours, France. 19-21 July 2019

This software includes a fully-parallel GPU implementation of the algorithm, as well as a CPU sequential version, provided for computational time comparison. Also we included a novel Luby-inspired greedy  coloring strategy, also implemented for GPU.


---
### Requirements
This software requires a Linux operating system (it has been developed and tested on Ubuntu 16.04). Both MCMC and Luby GPU implementations are targeted towards NVidia Graphic Processing Units. GPU portion of the code is written using the CUDA C library.

Hardware requirements are not strict, as any relatively modern x86-64 processor can compile and run the software. As the dataset is loaded on the host RAM in its entirety, having a decent amount of memory could come handy... However, graphs are stored in a so-called "compressed format", therefore even 2-4 GB of RAM would suffice for coloring moderately big graphs.

GPU requirements are also non-strict as this software requires a GPU with Compute Capability >= 3.5 and any recent CUDA SDK (>= 8.0) is supported. We tested this implementation on GPUs belonging to Kepler, Maxwell, Tesla and Pascal family.

Software wise, a recent C++ compiler is needed. GNU GCC (>= 5.4.0) is known to work, and we also tested the software with Intel C++ Compiler 2017 onward. A few additional packages are required, namely cmake (>= 2.8) and getopt: they both can be installed on Ubuntu Linux using the apt package manager:
```
sudo apt-get install cmake getopt
```
Other package managers (yum on CentOS / RedHat, for instance) provide a similar way to satisfy the library requirements.

---
### Downloading and compiling

Download the latest version from this page or clone the git repository altogether:

	git clone https://github.com/topopiccione/MCMC_Colorer.git

Once the package has been downloaded, move to the main directory, create a build dir, invoke cmake and build the software (`-j n` make option enables multi-thread compilation over n threads):

	cd MCMC_Colorer
	mkdir build
	cd build
	cmake ../src
	make -j 4

This will generate two executables: "MCMC_Colorer" and "datasetGen".

We additionally provide a few options to the makefile generation, namely a Compute Capability-specific code generation and compiling in debug mode.
| Option | Default | Description |
|-|-|-|
| -Dsm35 | OFF | Activates code generation for 3.5 Compute Capability NVidia GPUs |
| -Dsm50 | ON  | Activates code generation for 5.0 Compute Capability NVidia GPUs |
| -Dsm52 | OFF | Activates code generation for 5.2 Compute Capability NVidia GPUs |
| -Dsm60 | OFF | Activates code generation for 6.0 Compute Capability NVidia GPUs |
| -Dsm61 | OFF | Activates code generation for 6.1 Compute Capability NVidia GPUs |
| -Ddebug | OFF | Compile in debug mode |

All the options must be specified as cmake options, i.e. for compiling in debug mode for GPUs with Compute Capabilities 5.2 adn 6.1, disabling compiling for 5.0, the `cmake ../src` in the previous code snippet would become
```
cmake -Dsm50=OFF -Dsm52=ON -Dsm61=ON -Ddebug=ON ../src
```

For a quick test, launch the following command from the build directory:
```
./MCMC_Colorer --lubygpu --simulate 0.1 -n 1000
```
Results are saved in the 1000_0.100000_1.000000_out folder.

Also
```
./MCMC_Colorer --help
```
shows the help screen with the list of command line options.
![help_screen]

---

### Running MCMC colorer
MCMC_Colorer is a command line executable. At present time, no graphical user interface is provided. Its behavior is regulated by a set of command line options which are presented as follows.

#### Command line options
The following table summarizes all the available command line options:
| Category | Option | Mandatory | Default | Description |
|-|-|-|-|-|
|Dataset options| `--graph filename.txt` | Yes / No | - | Graph input file |
|| `--outDir name` | No | See descr. | Name of the output directory where coloring and logs will be saved. By default, is the name of the input file followed by "\_out"; in simulation mode, is "\<prob\>\_\<n\>_\<numColRatio\>\_out"  |
|| `--simulate p` | Yes / No | - | Launch the program on a randomly generated Erdos graph. Edges are generated with probability 0 \< p \< 1. In this modality, the `-n N`  option is mandatory |
|| `-n N` | Yes / No | - | Number of nodes of the randomly generated Erdos graph. Ignored when non in simulation mode |
|Coloring algorithm selection| `--mcmccpu` | (1) | - | Enables the MCMC CPU algorithm |
| | `--mcmcgpu` | (1) | - | Enables the MCMC GPU algorithm |
| | `--lubygpu` | (1) | - | Enables the Luby GPU algorithm |
|MCMC coloring options| `--nCol` | No | maxDeg / numColRatio | Choose the number of colors. By default is set to the maximum node degree divided by the `--numColRatio` value |
|| `--numColRatio` | No | 1.0 | Optional divider for the numer of colors `--nCol` |
|| `--tabooIterations` | No | 0 | Optional number of iteration for the taboo strategy |
|| `--tailcut` | No | No | Switch to greedy mode when few conflicts remain |
|General options| `--help` | No | - | Shows the help screen |
|| `--repet N` | No | 1 | Number of times each colorer algorithm is invoked for run  |
|| `--seed N` | No | Random | Integer seed for pseudo-random number generator  |

(1): more than one algorithm can be selected per run. If no selection is made, MCMC CPU is chosen by default.


---
#### Data format
Graph input file is a list of unidirected edges, with a single-line header, i.e.
```
single_line_header_can_be_whatever_you_want_who_cares
node1		node2		edgeWeight
node3		node12		edgeWeight
node4		node2		edgeWeight
node4		node5		edgeWeight
node6		node1		edgeWeight
node6		node8		edgeWeight
...
```
Data can be space / tab / comma separated valus. The single line header can be anything. The total number of nodes and edges are re-counted at import time, hence there is no need of specifying those in the header.  
The field "edgeWeight" is unused in the coloring, but the file importer supports its presence, as it is very common in publicly available datasets.  
We remark that edges in the list are supposed to be **unidirected**, hence for instance an edge from node 1 to node 6 implies that also node 6 is connected to node 1. **There is no need to specify the back edges in the input file, as the importer takes care of that. Also, duplicate / back edges may compromise the quality of solution or even the convergence of the algorithm.**

Each coloring algorithm provides two output files in the output directory: the coloring itself, i.e. a list of new-line-separated tuples of node_number / color_class and a log file providing basic statistics for nerds of the solution.

---
#### Random dataset generation
A random Erdos graph generator is included in this package as a way of testing the software without depending on external datasets. The random dataset generator is provided as a standalone application that generates a graph as a text file, and is also built-in the main MCMC Colorer executable.

The standalone application is called datasetGen and is invoked as following:
```
./datasetGen  number_of_nodes edge_probability output_filename
```
The output graph is saved in a format that is fully compatible with the main colorer executable.

Optionally, the random graph generator can be invoked from the main executable itself with the `--simulate` command line option, with the difference that the generated graph is not saved.

---
#### Examples
```
./MCMC_Colorer --lubygpu --simulate 0.1 -n 1000
```
Generates a random Erdos graph with 1000 nodes and 10% of probability of edges between each pair of nodes. Then the graph is colored with the Luby GPU algorithm.

```
./MCMC_Colorer --lubygpu --mcmccpu --mcmcgpu --simulate 0.1 -n 1000
```
As before, but the graph is colored with all the three algorithms

```
./MCMC_Colorer --mcmcgpu --mcmccpu --graph facebook.csv --repet 5
```
Import the graph from the file "facebook.csv" and colors it with both MCMC CPU and MCMC GPU algorithm. Each coloring is repeated 5 times. The number of colors is automatically set to the maximum node degree

```
./MCMC_Colorer --mcmcgpu --mcmccpu --graph facebook.csv --repet 5 --numColRatio 3.0
```
As the previous example, but the number of colors is automatically set to the maximum node degree / 3.0

```
./MCMC_Colorer --mcmcgpu --graph facebook.csv --nCol 500 --tailcut --out tempOutputDir
```
Import the graph from the file "facebook.csv" and colors it with MCMC GPU algorithm. The number of colors is set to 500. Tailcut greedy heuristic is enabled, hence the algorithm switches to a greedy strategy when very few conflicts remain. Results are saved in the "tempOutputDir" directory

---
### License

This package is distributed under the Apache 2.0 license. Please see the https://github.com/phuselab/MCMC_Colorer/LICENSE file for the complete version of the license.

Also, MCMC colorer uses several libraries whose source code is not included in the package, but it is automatically downloaded at compile time. These libraries are:

**Easylogging++**\
Zuhd Web Services\
(https://github.com/amrayn/easyloggingpp) \
Distributed under the MIT license\
Copy of the license is available at the project homepage





[banner]: https://raw.githubusercontent.com/Topopiccione/MCMC_Colorer/master/repo_pics/banner.png

[help_screen]: https://raw.githubusercontent.com/Topopiccione/MCMC_Colorer/master/repo_pics/help_screen.png
