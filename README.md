
# Softmux: Growing Digital Circuits with Deep Learning

This project explores _growing_ digital circuits using deep learning techniques. At its core is a learnable component called Softmux, capable of representing arbitrary _n_-input boolean functions. Softmux units take the place of traditional neurons where each unit learns a logic function.

During training, fuzzy bits — continuous values in the range [0, 1] — propagate through the network, enabling gradient-based optimization. After training, the network is discretized — replacing fuzzy bits with hard binary bits — resulting in a structure that can be directly translated into a digital circuit.

<div align="center">
  <img src="docs/images/digital-circuit-mnist.png" alt="Mockup of a digital circuit for handwritten digit recognition."/>
</div>

Early results on classic classification benchmarks show that the approach achieves competitive accuracy, which is largely retained even after full discretization.

## Related Work

[Neural Combinatorial Logic Circuit Synthesis from Input-Output Examples](https://github.com/pbelcak/neccs)

[Deep Differentiable Logic Gate Networks](https://github.com/Felix-Petersen/difflogic)

## Softmux Unit

A Softmux is a differentiable, logic-based computation unit capable of representing arbitrary _n_-input boolean functions. It behaves like a soft multiplexer, effectively acting as a trainable lookup table whose entries define the outputs of a logic function and are continuously optimized via gradient descent.

<div align="center">
  <img src="docs/images/softmux-unit.png" alt="A 2-input softmux unit."/>
</div>

Specifically, a 2-input Softmux takes two fuzzy binary inputs, S0 and S1, and has four trainable weights W = [W0, W1, W2, W3]. The weights are squashed into the [0, 1] range — Tk = σ(Wk) — using a smooth, bounded activation function (e.g.,  sigmoid). Each squashed weight, Tk, correspond to one entry in the truth table that defines the logic function. 

<div align="center">

| S0 | S1 | Selection   |
|----|----|-------------|
| 0  | 0  | T0          |
| 0  | 1  | T1          |
| 1  | 0  | T2          |
| 1  | 1  | T3          |

</div>

The output is an interpolation of the truth table entries, weighted by the selector inputs, producing a value that remains bounded within the [0, 1] range.

$$
Y = (1 − S0)(1 − S1)T0 +
    (1 − S0)(    S1)T1 +
    (    S0)(1 − S1)T2 + 
    (    S0)(    S1)T3
$$

This setup enables gradient-based optimization of logic behavior, making Softmux units compatible with standard deep learning pipelines while remaining interpretable as logic primitives. 

After training, the truth table entries are binarized into hard 0/1 logic, enabling a direct translation into logic gates.

<div align="center">

| Truth Table | Function    | Symbol                                            |
|-------------|-----------------|-------------------------------------------------|
| 0000        | FALSE           | ![FALSE](docs/images/gates/GATE-0000-FALSE.png) |
| 0001        | AND             | ![AND](docs/images/gates/GATE-0001-AND.png) |
| 0010        | A!B             | ![A!B](docs/images/gates/GATE-0010-AANDNOTB.png) |
| 0011        | A               | ![A](docs/images/gates/GATE-0011-A.png) |
| 0100        | !AB             | ![!AB](docs/images/gates/GATE-0100-NOTAANDB.png) |
| 0101        | B               | ![B](docs/images/gates/GATE-0101-B.png) |
| 0110        | XOR             | ![XOR](docs/images/gates/GATE-0110-XOR.png) |
| 0111        | OR              | ![OR](docs/images/gates/GATE-0111-OR.png) |
| 1000        | NOR             | ![NOR](docs/images/gates/GATE-1000-NOR.png) |
| 1001        | XNOR            | ![XNOR](docs/images/gates/GATE-1001-XNOR.png) |
| 1010        | !B              | ![!B](docs/images/gates/GATE-1010-NOTB.png) |
| 1011        | A+!B            | ![A+!B](docs/images/gates/GATE-1011-AORNOTB.png) |
| 1100        | !A              | ![!A](docs/images/gates/GATE-1100-NOTA.png) |
| 1101        | !A+B            | ![!A+B](docs/images/gates/GATE-1101-NOTAORB.png) |
| 1110        | NAND            | ![NAND](docs/images/gates/GATE-1110-NAND.png) |
| 1111        | TRUE            | ![TRUE](docs/images/gates/GATE-1111-TRUE.png) |

</div>

## Softmux Network
A Softmux Network is a sparse, multi-layer architecture. Each layer consists of a collection of Softmux units connected to a subset of outputs from previous layers. The sparsity emerges naturally, as the number of inputs per Softmux unit is significantly smaller than the total number of available outputs, which mimics traditional digital logic designs.

The final layer outputs multiple bits per class, with each bit acting as a voter. The votes are summed per class to produce class scores, forming a simple majority-voting mechanism. The class with the highest score is selected as the prediction.

<div align="center">
  <img src="docs/images/softmux-network.png" alt="Mockup of a softmux neural network."/>
</div>

## Implementation Details

Although the architecture can support general _n_-input logic functions and customizable connectivity, this work currently adopts a streamlined configuration: 2-input softmux units with a sigmoid squash function, layers with a fixed number of units, random connections between consecutive layers only.

## Training

The network is trained using standard gradient descent with CrossEntropyLoss applied to the class scores. The weights are initialized randomly within reasonable bounds, giving each unit an unbiased starting point from which it can adapt its logic function to fit the task.

A binary regularization term is applied to encourage each truth table entry to approach binary values. To allow flexible learning early on, binary regularization is deferred until the final 25% of epochs, giving the model time to explore smooth solutions before being pushed toward binary logic.

## Experiments

While the softmux network can be converted into a digital circuit, currently the focus is on validating the approach rather than hardware deployment. Therefore, all experiments were run using the discretized, software-based network, treating learned softmux units as fixed logic gates to evaluate classification performance.

Experiments were conducted on the symbolic MONK datasets and the binarized MNIST handwritten digit dataset where the grayscale images were converted to black-and-white images.

Training and inference were performed on an Apple M4 Pro Mac Mini using PyTorch on the Metal Performance Shaders (MPS) backend.

### Results

The softmux network achieved good performance across all evaluated datasets. 

On the MONK benchmarks, it reached 100% accuracy on MONK-1 (a clean, rule-based task). On MONK-2, it achieved 94.44% accuracy, and on MONK-3, 91.20% — both results comparable to or slightly below those of classical symbolic learners. 

On binarized MNIST, the model achieved 97.11% after discretization, not far behind the performance of conventional neural networks. 

<div align="center">

| <br>Dataset | <br>Layers | Units /<br>Layer | <br>Epochs | Training<br>(m) | Accuracy<br>(Fuzzy) | Accuracy<br>(Discrete) |
|---------|--------|---------------|--------|--------------|------------------|---------------------|
| MONK-1  | 6 |    24 | 250 | 0.24 | 100.00% | 100.00% |
| MONK-2  | 6 |    36 | 500 | 0.72  | 97.69% | 94.44% |
| MONK-3  | 6 |    36 | 500 | 0.49 | 91.67%  | 91.20% |
| MNIST   | 6 | 64000 | 200 | 319.2  | 97.26% | 97.11% |

</div>



These results suggest that a softmux network can learn expressive representations and retain accuracy even after full discretization.

### MNIST Dataset Result Analysis

The histogram below shows the distribution of logic functions across all layers of the network after discretization on the binarized MNIST dataset. 

<div align="center">
  <img src="docs/images/mnist-logic_function_histogram_grid.png" alt="Logic function histogram of the softmux network to predict the binarized MNIST handwritten digits." width=850/>
</div>

The high frequency of constant functions (FALSE and TRUE) indicates significant redundancy in the network, suggesting it could be greatly simplified. Additionally, the frequent use of pass-through functions (A and B) may imply that the network is implicitly learning skip connections by propagating information directly through layers.

Notably, the heat map reveals a striking symmetry in logic function usage — a pattern that did not appear in the MONK benchmarks. At this stage, no clear explanation has been proposed to account for this structural bias.

To rule out simple memorization, the network was also validated using newly drawn handwritten digits on a virtual canvas. It correctly classified the vast majority of these inputs, providing empirical evidence that it had generalized beyond the training data.

However, the network could still be fooled in a few edge cases — for example, an “8” with an unusually large lower loop was consistently misclassified as a “0”, and a “3” with an extended top stroke was often interpreted as a “7”. Such errors are not uncommon also in conventional neural networks.

## Remarks and Future Directions
### Logic Synthesis & Optimization
Once the Softmux network is trained and discretized, the resulting logic circuit can be exported to formats like BLIF or Verilog and optimized using tools such as ABC. This enables logic - level transformations—like structural hashing, minimization, and LUT mapping — that reduce gate count and circuit depth, making the network more efficient and hardware-friendly.

Comparing pre and post-synthesis metrics (e.g., gate count, depth) can provide insight into redundancy and structural patterns learned during training. 
