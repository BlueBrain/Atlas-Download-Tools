Synchronization
===============
One of the main goals of ``atldld`` is to replicate some of the behavior of
`Allen Brain Synchronization <https://help.brain-map.org/display/api/Image-to-Image+Synchronization>`_
locally. This in turn allows for synchronization of a large number of coordinates in a
reasonable amount of time. The synchronization logic is implemented in the
``atldl.sync`` module.

Reference-To-Image
------------------
`Reference-To-Image API
<http://help.brain-map.org/display/api/Image-to-Image+Synchronization#Image-to-ImageSynchronization-Reference-To-Image>`_

For each dataset the Allen Brain Institute provides a matrix :math:`A_{trv}` that encodes a 3D affine
transformation.

.. math::

	A_{trv} = \begin{pmatrix}
	 a_{1,1} & a_{1,2} & a_{1,3} & a_{1,4}\\ 
	 a_{2,1} & a_{2,2} & a_{2,3} & a_{2,4}\\
	 a_{3,1} & a_{3,2} & a_{3,3} & a_{3,4}\\
	\end{pmatrix}

Additionally, each section image has a matrix :math:`B_{tvs}` that encodes a 2D affine
transformation.

.. math::

	B_{tvs} = \begin{pmatrix}
	 b_{1,1} & b_{1,2} & b_{1,3}\\ 
	 b_{2,1} & b_{2,2} & b_{2,3}\\
	\end{pmatrix}

They can be used to map any point :math:`(p, i, r)` from the reference space to
a corresponding point :math:`(x, y)` in the image space.
Specifically, the exact mapping looks like this

.. math::

	\begin{pmatrix}
	x\\ 
	y\\ 
	s\\
	\end{pmatrix}
	=
	\begin{pmatrix}
	 b_{1,1} & b_{1,2} & 0 & b_{1,3}\\ 
	 b_{2,1} & b_{2,2} & 0 & b_{2,3}\\
	 0 & 0 & 1 & 0\\ 
	\end{pmatrix}
	 \begin{pmatrix}
	 a_{1,1} & a_{1,2} & a_{1,3} & a_{1,4}\\ 
	 a_{2,1} & a_{2,2} & a_{2,3} & a_{2,4}\\
	 a_{3,1} & a_{3,2} & a_{3,3} & a_{3,4}\\
	 0 & 0 & 0 & 1\\
	\end{pmatrix}
	\begin{pmatrix}
	p\\ 
	i\\ 
	r\\
        1\\
	\end{pmatrix}


Note that :math:`s` represents the theoretical section number multiplied
by the dataset thickness.

The above described logic is implemented inside of ``atldl.sync.pir_to_xy``.
See below a small example.


.. testcode::

   from atldld.sync import pir_to_xy
   import numpy as np

   affine_2d = np.array(
       [
           [1, 0, 0],
           [0, 1, 0],
       ]
   )  # should be downloaded from the Allen Brain API
   affine_3d = np.array(
        [
           [1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 1, 0],
        ]
   )  # should be downloaded from the Allen Brain API

   coords_ref = np.array(
        [
              [2, 1],
              [5, 2],
              [7, 5],
        ]
   )  # (3, n_coords)

   coords_img = pir_to_xy(coords_ref, affine_2d, affine_3d)

   print(coords_img)

.. testoutput::

    [[ 2.  1.]
     [10.  4.]
     [ 7.  5.]]

Note that the shape of ``coords_ref`` is ``(3, n_coords=2)``. However, ``n_coords``
can be much larger and in fact the user is encouraged to make it as large
as possible to get a speedup.




Image-To-Reference
------------------
`Image-To-Reference API
<https://help.brain-map.org/display/api/Image-to-Image+Synchronization#Image-to-ImageSynchronization-Image-To-Reference>`_

For each dataset the Allen Brain Institute provides a matrix :math:`A_{tvr}` that encodes a 3D affine
transformation.

.. math::

	A_{tvr} = \begin{pmatrix}
	 a_{1,1} & a_{1,2} & a_{1,3} & a_{1,4}\\ 
	 a_{2,1} & a_{2,2} & a_{2,3} & a_{2,4}\\
	 a_{3,1} & a_{3,2} & a_{3,3} & a_{3,4}\\
	\end{pmatrix}

Additionally, each section image has a matrix :math:`B_{tsv}` that encodes a 2D affine
transformation.

.. math::

	B_{tsv} = \begin{pmatrix}
	 b_{1,1} & b_{1,2} & b_{1,3}\\ 
	 b_{2,1} & b_{2,2} & b_{2,3}\\
	\end{pmatrix}

They can be used to map any point :math:`(x, y)` from the image space to
a corresponding point :math:`(p, i, r)` in the reference space.
Specifically, the exact mapping looks like this

.. math::

	\begin{pmatrix}
	p\\ 
	i\\ 
	r\\
	\end{pmatrix}
	=
	 \begin{pmatrix}
	 a_{1,1} & a_{1,2} & a_{1,3} & a_{1,4}\\ 
	 a_{2,1} & a_{2,2} & a_{2,3} & a_{2,4}\\
	 a_{3,1} & a_{3,2} & a_{3,3} & a_{3,4}\\
	\end{pmatrix}
	\begin{pmatrix}
	 b_{1,1} & b_{1,2} & 0 & b_{1,3}\\ 
	 b_{2,1} & b_{2,2} & 0 & b_{2,3}\\
	 0 & 0 & 1 & 0\\ 
	 0 & 0 & 0 & 1\\
	\end{pmatrix}
	\begin{pmatrix}
	x\\ 
	y\\ 
        s\\
        1\\
	\end{pmatrix}


Note that :math:`s` is a product of the section thickness (dataset specific)
and the section number (image specific). Both of these values can
be retrieved from the Allen Brain API.

The above described logic is implemented inside of ``atldl.sync.xy_to_pir``.
See below a small example.


.. testcode::

   from atldld.sync import xy_to_pir
   import numpy as np

   affine_2d = np.array(
       [
           [1, 0, 0],
           [0, 3, 0],
       ]
   )  # should be downloaded from the Allen Brain API
   affine_3d = np.array(
        [
           [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
        ]
   )  # should be downloaded from the Allen Brain API

   coords_img = np.array(
        [
              [2, 1, 5, 11, 0, 15],
              [5, 2, 10, 12, 0, 2],
              [7, 5, 2, 21, 0, 0],
        ]
   )  # (3, n_coords)

   coords_ref = xy_to_pir(coords_img, affine_2d, affine_3d)

   print(coords_ref)

.. testoutput::

    [[ 2.  1.  5. 11.  0. 15.]
     [15.  6. 30. 36.  0.  6.]
     [ 7.  5.  2. 21.  0.  0.]]

Note that the shape of ``coords_img`` is ``(3, n_coords=6)``. However, ``n_coords``
can be much larger and in fact the user is encouraged to make it as large
as possible to get a speedup.

