import numpy as np
from numpy.linalg import norm, svd
from numpy.linalg import norm
from scipy.linalg import qr
import cv2
import os
from scipy.io import loadmat, savemat
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
from scipy import misc

d1 = 200
d2 = 200
batch = 10

def rgb2gray(rgb):
	r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray

def inexact_augmented_lagrange_multiplier(X, lmbda=.01, tol=1e-3,
                                          maxiter=100, verbose=True):
    """
    Inexact Augmented Lagrange Multiplier
    """
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[0]
    itr = 0
    while True:
        Eraw = X - A + (1 / mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + \
                             np.minimum(Eraw + lmbda / mu, 0)
        U, S, V = svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        Aupdate = np.dot(
            np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        A = Aupdate
        E = Eupdate
        Z = X - A - E
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / dnorm) < tol) or (itr >= maxiter):
            break
    if verbose:
        print("Finished at iteration %d" % (itr))
    return A, E


def wthresh(a, thresh):
    # Soft wavelet threshold
    res = np.abs(a) - thresh
    return np.sign(a) * ((res > 0) * res)

# Default threshold of .03 is assumed to be for input in the range 0-1...
# original matlab had 8 out of 255, which is about .03 scaled to 0-1 range


def go_dec(X, thresh=.03, rank=2, power=0, tol=1e-3,
           max_iter=100, random_seed=0, verbose=True):
    m, n = X.shape
    if m < n:
        X = X.T
    m, n = X.shape
    L = X
    S = np.zeros(L.shape)
    itr = 0
    random_state = np.random.RandomState(random_seed)
    while True:
        Y2 = random_state.randn(n, rank)
        for i in range(power + 1):
            Y1 = np.dot(L, Y2)
            Y2 = np.dot(L.T, Y1);
        Q, R = qr(Y2, mode='economic')
        L_new = np.dot(np.dot(L, Q), Q.T)
        T = L - L_new + S
        L = L_new
        S = wthresh(T, thresh)
        T -= S
        err = norm(T.ravel(), 2)
        if (err < tol) or (itr >= max_iter):
            break
        L += T
        itr += 1
    # Is this even useful in soft GoDec? May be a display issue...
    G = X - L - S
    if m < n:
        L = L.T
        S = S.T
        G = G.T
    if verbose:
        print("Finished at iteration %d" % (itr))
    return L, S, G


def read_img_sequence(dir_path):
	print "begin reading "

	X = np.zeros((d1, d2, batch), dtype=np.uint8)
	test_match = None
	for i, image_path in enumerate(os.listdir(dir_path)):
		
		img_abs_path = dir_path + os.sep + image_path
		print "reading img:%s"%img_abs_path
		img = cv2.resize(cv2.imread(img_abs_path,cv2.IMREAD_GRAYSCALE).astype(np.uint8), (d1,d2))
	
		X[:, :, i] = img
	X = X.reshape((d1 * d2, batch))
	print "read end"
	return X
# demo inspired by / stolen from @kuantkid on Github - nice work!


def mlabdefaults():
    matplotlib.rcParams['lines.linewidth'] = 1.5
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams['font.size'] = 22
    matplotlib.rcParams['font.family'] = "Times New Roman"
    matplotlib.rcParams['legend.fontsize'] = "small"
    matplotlib.rcParams['legend.fancybox'] = True
    matplotlib.rcParams['lines.markersize'] = 10
    matplotlib.rcParams['figure.figsize'] = 8, 5.6
    matplotlib.rcParams['legend.labelspacing'] = 0.1
    matplotlib.rcParams['legend.borderpad'] = 0.1
    matplotlib.rcParams['legend.borderaxespad'] = 0.2
    matplotlib.rcParams['font.monospace'] = "Courier New"
    matplotlib.rcParams['savefig.dpi'] = 200




def make_video(X, alg, cache_path='D:/work/face_expression/test_feature'):

	name = alg
	print cache_path, os.path.exists(cache_path)
	if not os.path.exists(cache_path):
		os.mkdir(cache_path)
	# If you generate a big
	if not os.path.exists('%s/%s_tmp' % (cache_path, name)):
		os.mkdir("%s/%s_tmp" % (cache_path, name))
	mat = loadmat('./%s_background_subtraction.mat'%(name))
	org = X.reshape(d1, d2, X.shape[1])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	usable = [x for x in sorted(mat.keys()) if "_" not in x][0]
	sz = min(org.shape[2], mat[usable].shape[2])
	print "sz",sz
	for i in range(sz):
		ax.cla()
		ax.axis("off")
		ax.imshow(np.hstack([mat[x][:, :, i] for x in sorted(mat.keys()) if "_" not in x] + [org[:, :, i]]), cm.gray)
		fname_ = '%s/%s_tmp/_tmp%03d.png'%(cache_path, name, i)
		if (i % 25) == 0:
			print('Completed frame', i, 'of', sz, 'for method', name)
		fig.tight_layout()
		fig.savefig(fname_, bbox_inches="tight")
	# Write out an mp4 and webm video from the png files. -r 5 means 5 frames a second
	# libx264 is h.264 encoding, -s 160x130 is the image size
	# You may need to sudo apt-get install libavcodec
	plt.close()

	num_arrays = na = len([x for x  in mat.keys() if "_" not in x])
	cdims = (na * d1, d2)
	cmd_h264 = "ffmpeg -i %s/%s_tmp/_tmp%%03d.png -c:v libx264 -r 10  -crf 25 -s 1470x640 -pix_fmt yuv420p"%(cache_path,name) +" -s %dx%d -preset ultrafast %s_animation.mp4 -y"% (cdims[0], cdims[1], name)
	#cmd_h264 = "ffmpeg -y -r 10 -i '%s/%s_tmp/_tmp%%03d.png' -c:v libx264 " % (cache_path, name) + "-s %dx%d -preset ultrafast -pix_fmt yuv420p %s_animation.mp4" % (cdims[0], cdims[1], name)
	#cmd_vp8 = "ffmpeg -y -r 10 -i '%s/%s_tmp/_tmp%%03d.png' -c:v libvpx " % (cache_path, name) + "-s %dx%d -preset ultrafast -pix_fmt yuv420p %s_animation.webm" % (cdims[0], cdims[1], name)
	print cmd_h264
	os.system(cmd_h264)
	#os.system(cmd_vp8)


def make_matfile(godec=False, rpca=False):
	print 1
	X = read_img_sequence("D:/work/single_depth/godec")
	print 2

	if rpca:
	
		L,S = inexact_augmented_lagrange_multiplier(X)
		L = L.reshape(d1, d2, batch)
		S = S.reshape(d1, d2, batch)
		O = X.reshape(d1,d2,batch)
		for frame in range(batch):
			Limg = L[:,:,frame]
			Simg = S[:,:,frame]
			orig = O[:,:,frame]
			cv2.imshow("S",Simg)
			cv2.imshow("L",Limg)
			cv2.imshow("orig",orig)
			cv2.waitKey(0)

	if godec:
		L, S, G = go_dec(X)
		O = X.reshape(d1,d2,batch)

		L = L.reshape(d1, d2, batch)
		S = S.reshape(d1, d2, batch)
		G = G.reshape(d1, d2, batch)
		for frame in range(batch):
			Simg = S[:,:,frame]
			Limg = L[:,:,frame]
			Gimg = G[:,:,frame]
			Oimg = O[:,:,frame]
			#print img.shape
			cv2.imshow("S", Simg)
			cv2.imshow("L", Limg)
			cv2.imshow("G", Gimg)
			cv2.imshow("O", Oimg)
			cv2.waitKey(0)
		savemat("D:/work/face_expression/test_feature/GoDec_background_subtraction.mat", {"1": L, "2": S, "3": G, })
		print("GoDec complete")
	return X

if __name__ == "__main__":
	X = make_matfile(godec=True,rpca=False)
	'''
	mlabdefaults()
	all_methods = ['GoDec']
	for name in all_methods:
		make_video(X, name);
	'''

