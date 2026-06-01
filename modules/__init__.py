# modules/__init__.py
from .basic_ops import to_grayscale, to_binary, arithmetic_operation, logical_operation
from .histogram import plot_histogram
from .convolution import apply_filter, get_kernel_display, list_filters
from .morphology import apply_morphology, get_se_display, list_se, list_operations
