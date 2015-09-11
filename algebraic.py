""" Functions for algebraic fitting """
import numpy as np


def to_cartesian(r, theta, center = (0, 0)):
    y = r * np.sin(theta) + center[0]
    x = r * np.cos(theta) + center[1]
    return y, x


def to_radial(y, x, center=(0, 0)):
    yc = y - center[0]
    xc = x - center[1]    
    r = np.sqrt(yc**2 + xc**2)
    theta = np.arctan2(yc, xc)
    return r, theta

def gauss(x, *p):
    """
    (not normalized) gaussian, for fitting purposes.
    A * sigma * sqrt(2 pi) gives area
    """
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
def ellipse_perimeter(a, b):
    """Approximation by Ramanujan"""
    h = ((a - b)**2)/((a + b)**2)
    return np.pi*(a+b)*(1 + 3*h/(10 + np.sqrt(4 - 3*h)))

def fitEllipse(data):
    """ Fits an ellipse algebraically to datapoints, using an eigenvalue
    equation.

    Parameters
    ----------
    data : numpy array of floats
        array of shape (N, 2) containing datapoints

    Returns
    -------
    r1, r2, x, y, angle
        (r1, r2) the two pricinple radii of the ellipse
        (x, y) coordinates of ellipse center
        angle of r1 with the x-axis, in radians

    See also
    --------
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """
    x = data[:, 0][:, np.newaxis]
    y = data[:, 1][:, np.newaxis]
    D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    p = V[:, n]

    a, b, c, d, f, g = p[0], p[1]/2, p[2], p[3]/2, p[4]/2, p[5]
    num = b*b - a*c
    x0 = (c*d - b*f)/num
    y0 = (a*f - b*d)/num
    up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
    down1 = (b*b - a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c))) - (c+a))
    down2 = (b*b - a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c))) - (c+a))
    res1 = np.sqrt(up/down1)
    res2 = np.sqrt(up/down2)
    angle = 0.5*np.arctan(2*b/(a-c))
    return res1, res2, x0, y0, angle


def fitEllipseStraight(data):
    """ Fits an ellipse algebraically to datapoints, using an eigenvalue
    equation. The principle axes are forced on the x, y axes.

    Parameters
    ----------
    data : numpy array of floats
        array of shape (N, 2) containing datapoints

    Returns
    -------
    xr, yr, xc, yc
        (xr, yr) the two pricinple radii of the ellipse
        (xc, yc) coordinates of ellipse center

    See also
    --------
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """
    x = data[:, 0][:, np.newaxis]
    y = data[:, 1][:, np.newaxis]
    D = np.hstack((x*x, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([5, 5])
    C[0, 1] = C[1, 0] = 1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    p = V[:, n]

    a, c, d, f, g = p[0], p[1], p[2]/2, p[3]/2, p[4]
    xc = -d/a
    yc = -f/c
    num = a*f**2 + c*d**2 - a*c*g
    denom1 = a**2*c
    denom2 = a*c**2
    xr = np.sqrt(num/denom1)
    yr = np.sqrt(num/denom2)
    return xr, yr, xc, yc


def drawEllipse(p, alpha=0, n=100, angle_range=(0, 2*np.pi)):
    """ Calculates points on an ellipse

    Parameters
    ----------
    p : tuple of float
        (r1, r2, x, y) the two pricinple radii of the ellipse
    angle : float
        angle of r1 with the x-axis, in radians
    n : int
        number of points
    angle_range : tuple of float

    Returns
    -------
    data : numpy array of floats
        array of shape (N, 2) containing datapoints
    """
    r1, r2, xc, yc = p
    R = np.array([
        [np.cos(alpha), np.sin(alpha)],
        [-np.sin(alpha), np.cos(alpha)]
    ])
    a0, a1 = angle_range
    angles = np.linspace(a0, a1, n)
    X = np.vstack([np.cos(angles) * r1,
                   np.sin(angles) * r2]).T
    return np.dot(X, R) + (xc, yc)


def fit_circle(features):
    # from x, y points, returns an algebraic fit of a circle 
    # (not optimal least squares, but very fast)
    # returns center, radius and rms deviation from fitted

    x = features[:,1]
    y = features[:,0]
    
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    
    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center in reduced coordinates (uc, vc):
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    try:
        uc, vc = np.linalg.solve(A, B)
    except:
        return (0, 0), 0, 9999

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calculation of all distances from the center (xc_1, yc_1)
    Ri_1      = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1       = np.mean(Ri_1)
    sqrdeviations = (Ri_1-R_1)**2
  #  wanted = np.argsort(deviations)[len(x) // 8:len(x)*7 // 8]
  #  residu_filt = np.sum(deviations[wanted])
    #residu2_1 = np.sum((Ri_1**2-R_1**2)**2)
    #print residu_1, residu2_1
    return (yc_1, xc_1), R_1, sqrdeviations