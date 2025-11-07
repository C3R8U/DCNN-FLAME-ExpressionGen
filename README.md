# DCNN-FLAME-ExpressionGen
Overview

DCNN-FLAME-ExpressionGen is an official reference implementation of the Dual-Branch Convolutional Neural Network with FLAME-based Parametric Supervision (DCNN-FLAME) model for expressive facial animation and emotion-driven avatar synthesis. This framework integrates a multi-scale convolutional backbone, identity-preserving encoder, expression classification head, and Action Unit (AU) detector with FLAME-inspired geometric parameter regression. It is designed to generate realistic and emotionally consistent facial expressions for animated characters and stylized human faces.

Features

The model combines dual-branch supervision—expression recognition and AU prediction—with multi-scale feature fusion to enhance fine-grained expression control. It employs perceptual (VGG19-based) and Maximum Mean Discrepancy (MMD) losses for domain adaptation between human and animated face datasets. The architecture is lightweight (≈47M parameters) and optimized for real-time inference (~27 FPS on RTX 3090 GPU).

Repository Structure

dcnn_flame.py — Core model definition and loss computation

train_flame.py — End-to-end training and validation script
