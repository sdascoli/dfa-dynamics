import copy
import torch
import torch.nn as nn

from .dfa import DFA, DFALayer

class AlignmentMeasurement:
    def __init__(self, model, bp_device, sensitivity=1e-8, bp_model=None):
        # Browse through DFA model and find the backend:
        self.dfa_model = model
        self.dfa_backend = None
        for module in self.dfa_model.modules():
            if isinstance(module, DFA):
                self.dfa_backend = module

        # Build the BP model from the DFA one, disable DFA features, and find the backend:
        self.bp_device = bp_device
        if bp_model is None:
            self.bp_model = copy.deepcopy(self.dfa_model).to(bp_device)
        else:
            self.bp_model = bp_model
        self.bp_backend = None
        for module in self.bp_model.modules():
            if isinstance(module, DFA):
                module.no_training = True  # Disable random projection at feedback points.
                module.rp_device = bp_device
                self.bp_backend = module

        for dfa_layer in self.bp_backend.dfa_layers:
            dfa_layer.passthrough = True  # Let the gradients flow freely.

        self.dfa_gradients = {}
        self.bp_gradients = {}
        self.hooks_registry = []

        self.sensitivity = sensitivity
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=self.sensitivity)

    def measure_alignment(self, input, target, loss_function, return_difference=False):
        self.enable_alignment_measurement()
        self.bp_model.load_state_dict(self.dfa_model.state_dict())

        output_dfa = self.dfa_model(input)
        output_bp = self.bp_model(input.to(self.bp_device))

        loss_dfa = loss_function(output_dfa, target)
        loss_bp = loss_function(output_bp, target.to(self.bp_device))

        loss_dfa.backward()
        loss_bp.backward()
        self.disable_alignment_measurement()

        alignments = {'layers':[]}
        grads_bp = []
        grads_dfa = []
        for l, (dfa_module, bp_module) in enumerate(zip(self.dfa_model.modules(), self.bp_model.modules())):
            if len(list(dfa_module.modules())) == 1:
                if not isinstance(dfa_module, DFALayer):
                    if dfa_module not in self.dfa_gradients:
                        print(f"WARNING! Module {dfa_module} has not been found in the gradients! "
                              f"It has received no updates in training. Is that expected? Check your gradient flow.")
                    else:
                        grad_dfa = self.dfa_gradients[dfa_module]
                        grad_bp = self.bp_gradients[bp_module]
                        grad_dfa[grad_dfa.abs() <= self.sensitivity] = 0
                        grad_bp[grad_bp.abs() <= self.sensitivity] = 0
                        grad_dfa = grad_dfa.contiguous().view(grad_dfa.shape[0], -1) / self.sensitivity  # Condition the gradients to avoid small values
                        grad_bp = grad_bp.contiguous().view(grad_bp.shape[0], -1) / self.sensitivity
                        grads_dfa.append(grad_dfa)
                        grads_bp.append(grad_bp)
                        angle = self.cosine_similarity(grad_bp.to(grad_dfa.device), grad_dfa)
                        angle[grad_dfa.sum(dim=1) + grad_bp.sum(dim=1).to(grad_dfa.device) == 0] = 1.
                        alignment = float(angle.mean())
                        if isinstance(dfa_module, nn.Conv2d) or isinstance(dfa_module, nn.Linear):
                            alignments['layers'].append(alignment)
        alignments['layers'] = alignments['layers'][:-1] #ignore last layer     
        grads_bp = torch.cat(grads_bp[:-2], dim=1)
        grads_dfa = torch.cat(grads_dfa[:-2], dim=1)
        angle = self.cosine_similarity(grads_bp.to(grad_dfa.device), grads_dfa)
        angle[grads_dfa.sum(dim=1) + grads_bp.sum(dim=1).to(grads_dfa.device) == 0] = 1.
        alignment = float(angle.mean())
        alignments['total'] = alignment
        self.dfa_gradients = {}
        self.bp_gradients = {}

        if return_deviation:
            difference = None
        else:
            difference = grads_bp - grads_dfa
        
        return alignments, difference

    def enable_alignment_measurement(self):
        for i, module in enumerate(self.dfa_model.modules()):
            if len(list(module.modules())) == 1:
                self.hooks_registry.append(module.register_backward_hook(self.dfa_hook))

        for module in self.bp_model.modules():
            if len(list(module.modules())) == 1:
                if not isinstance(module, DFALayer):
                        self.hooks_registry.append(module.register_backward_hook(self.bp_hook))

    def disable_alignment_measurement(self):
        for hook in self.hooks_registry:
            hook.remove()
        self.hooks_registry = []

    def dfa_hook(self, module, grad_input, grad_output):
        self.dfa_gradients[module] = grad_output[0]

    def bp_hook(self, module, grad_input, grad_output):
        self.bp_gradients[module] = grad_output[0]
