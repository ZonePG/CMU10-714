# %% [markdown]
# <a href="https://colab.research.google.com/github/dlsyscourse/lecture5/blob/main/5_automatic_differentiation_implementation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Lecture 5: Automatic Differentiation Framework Implementation
# 
# In this lecture, we will walk through the design elements of an automatic differentiation framework.
# 
# We won't implement the automatic differentiation(that is part of your homework), but we will walk through all the scaffolds that build up our overall framework.
# 
# 
# 

# %% [markdown]
# ## Prepare the codebase
# 
# To get started, we can clone the following repo from the github.

# %%
# Code to set up the assignment
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/
# !mkdir -p 10714f23
# %cd /content/drive/MyDrive/10714f23
# NOTE: Run the following line
# - uncomment the following line if you run this section for the first time
# - comment and skip the following line when you run this section for a second time
#   so you will have a local copy of lecture5 under 10714f22/lecture5 that you can
#   continue to edit and play with
# !git clone https://github.com/dlsyscourse/lecture5
# !ln -s /content/drive/MyDrive/10714f23/lecture5 /content/needle

# %% [markdown]
# We can then run the following command to make the path to the package available in colab's environment as well as the PYTHONPATH.

# %%
# %set_env PYTHONPATH /content/needle/python:/env/python
import sys
sys.path.append("/Users/zoupeng/code/CMU10-714/lectures/lecture5/python")

# %% [markdown]
# ## Needle codebase walkthrough
# 
# Now click the files panel on the left side. You should be able to see the following files:
# 
# - `needle/python/needle`
#     - `__init__.py`
#     - `auograd.py`
#     - `backend_numpy.py`
#     - `ops/`
# 
# Our framework is called needle. Needle stands for **ne**cessary **e**lements of **d**eep **le**arning.
# You can also viewed it as a sewing needle that threads through clothes
# to form (neural)net patterns, and the create traces for automatic differentiation.
# 
# 
# 

# %% [markdown]

# %% [markdown]
# ### Tensor creation and manipulation
# 
# 
# 

# %%
import needle as ndl

# %% [markdown]
# We can call perform array operations on needle Tensors. The following code creates a new Tensor y by adding a constant scalar to x.

# %%
x = ndl.Tensor([1,2,3], dtype="float32")

# %%
x

# %%


# %%


# %% [markdown]
# We provide common operator overloadingss so you can also directly write + and the call redirects to ndl.add_scalar.

# %%
y = x + 1

# %%
y

# %% [markdown]
# We can call `y.numpy()` to explicitly convert a needle Tensor to a numpy ndarray. Note for numpy backend, needle tensor is backed by an numpy.ndarray so there isn't much a difference here. However, we will implement other non-numpy backends that are backed by different kind of devices.

# %%
y.numpy()

# %% [markdown]
# ### Key Data Structures
# 
# Needle contains the following key data structures:
# 
# - Value: represent a value "node" in a computational graph
#     - Tensor is a subclass of Value.
#     - We might introduce other kinds of Value in the future (e.g. tuple of tensors)
# - Op: represent the kind of computation we perform at each node.
# 
# 

# %% [markdown]
# ## Computational Graph
# 
# When running array computations, needle not only executes the arithmetic operations, but also creates a computational graph along the way.

# %% [markdown]

# %%


# %%


# %%


# %%


# %%
def print_node(node):
    print("id=%d" % id(node))
    print("inputs=%s" % [id(x) for x in node.inputs])
    print("op=%s" % type(node.op))
    print("data=%s" % node.cached_data)

# %% [markdown]
#  NOTE: we need to open up the ops and add implementation of exp in order for the following code to run
# 
# 

# %%
v1 = ndl.Tensor([0], dtype="float32")
v2 = ndl.exp(v1)
v3 = v2 + 1
v4 = v2 * v3

# %%
v2adj, v3adj = v4.op.gradient_as_tuple(
    ndl.Tensor([1], dtype="float32"),
    v4,
)

# %%
print_node(v2adj)

# %%


# %%


# %% [markdown]
# Each op have a gradient function, that defines how to propagate adjoint back into its inputs(as partial adjoints). We can look up the gradient implementation of v4.op as follows(impl of gradient for multiplication)
# 

# %%
print(v4.op.gradient.__code__)

# %%
v4.op

# %% [markdown]
# The gradient function defines a single step to propagate the output adjoint to partial adjoints of its inputs.

# %% [markdown]
# ## Additional Remarks
# 
# While needle is designed as a minimalist framework, it contains a comprehensive the bells and whistles of standard deep learning frameworks.
# 
# - Read and think about the relation between Tensor, array_api and underlying NDArray.
# - Think about how gradient are implemented.


