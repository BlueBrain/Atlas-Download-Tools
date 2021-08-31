Synchronization
===============
One of the main goals of ``atldld`` is to replicate some of the behavior of
`Allen Brain Synchronization <https://help.brain-map.org/display/api/Image-to-Image+Synchronization>`_
locally. This in turn allows for synchronization of a large number of coordinates in a
reasonable amount of time..

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

	B_{tvr} = \begin{pmatrix}
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
and the section number (image specific).
