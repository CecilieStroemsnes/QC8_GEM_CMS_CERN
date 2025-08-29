# geometry.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

class ME0_Geometry:
    """
    Geometry of the ME0 GEM detector stack and scintillators.
    All dimensions in meters.

    Attributes:
        bottom_width (float): Width of the GEM trapezoid at the bottom (m).
        top_width (float): Width of the GEM trapezoid at the top (m).
        total_length (float): Total length of the GEM trapezoid (m).
        rect_length (float): Length of the rectangular section at the top of the GEM trapezoid (m).
        layer_spacing (float): Spacing between GEM layers (m).
        n_layers (int): Number of GEM layers in the stack.
        scintillator_width (float): Width of the scintillators (m).
        scintillator_length (float): Length of the scintillators (m).
        bottom_gap (float): Gap between the bottom scintillator and the GEM stack (m).
        top_gap (float): Gap between the top scintillator and the GEM stack (m).
    """
    def __init__(self,
                 position = "low",
                 bottom_width=0.2352,
                 top_width=0.460,
                 total_length=0.7879,
                 rect_length=0.150,
                 layer_spacing=0.0352,
                 n_layers=6,
                 scintillator_width=0.400,
                 scintillator_length=0.800,
                 ):

        # Position: "low" or "high" (scintillator placement)
        pos = str(position).lower().strip()
        if pos == "low":
            bottom_gap, top_gap = 0.20, 0.45
        elif pos == "high":
            bottom_gap, top_gap = 0.45, 0.20
        else:
            raise ValueError("position must be 'low' or 'high'")
        
        # Initialize ME0 geometry with given parameters.
        self.bottom_width = bottom_width
        self.top_width = top_width
        self.total_length = total_length
        self.rect_length = rect_length
        self.layer_spacing = layer_spacing
        self.n_layers = n_layers
        self.scintillator_width = scintillator_width
        self.scintillator_length = scintillator_length
        self.bottom_gap = bottom_gap
        self.top_gap = top_gap

        # -----------------------------------------------------------
        # --------------- ME0 geometry ------------------------------
        # -----------------------------------------------------------

        # GEM trapezoid outline (top view)
        self.x_top = [
            -top_width / 2,
             top_width / 2,
             top_width / 2,
             bottom_width / 2,
            -bottom_width / 2,
            -top_width / 2,
            -top_width / 2
        ]
        self.y_top = [
            total_length,
            total_length,
            total_length - rect_length,
            0,
            0,
            total_length - rect_length,
            total_length
        ]

        # Create a Path object for the GEM trapezoid
        self.gem_path = Path(np.column_stack((self.x_top, self.y_top))) 

        # Z positions of GEM layers
        self.z_bottom_stack = 0.0 # Bottom layer z-coordinate
        self.z_top_stack = (n_layers - 1) * layer_spacing # Top layer z-coordinate
        self.layer_z = np.array([i * layer_spacing for i in range(n_layers)]) # Z-coordinates of all layers

        # -----------------------------------------------------------
        # --------------- Scintillator geometry ----------------------
        # -----------------------------------------------------------

        # Scintillator bounds (top view)
        self.scin_xmin = -scintillator_width/2
        self.scin_xmax = scintillator_width/2
        self.scin_ymin = 0.0
        self.scin_ymax = scintillator_length

        # Scintillator z positions for top and bottom
        self.Z_TOP_SCIN = self.z_top_stack + self.top_gap
        self.Z_BOTTOM_SCIN = self.z_bottom_stack - self.bottom_gap

        # Z-coordinates of the centers of top and bottom scintillators
        self.scin1_center = self.Z_BOTTOM_SCIN - scintillator_length/2
        self.scin2_center = self.Z_TOP_SCIN + scintillator_length/2

        # Scintillator outlines (side view)
        self.scin1_x, self.scin1_y = self.rect_xy(scintillator_width, self.scin1_center, scintillator_length)
        self.scin2_x, self.scin2_y = self.rect_xy(scintillator_width, self.scin2_center, scintillator_length)

        # Scintillator outline (top view)
        self.scin_x_top = [-scintillator_width/2, scintillator_width/2,
                           scintillator_width/2, -scintillator_width/2,
                           -scintillator_width/2]
        self.scin_y_top = [0, 0,
                           scintillator_length, scintillator_length,
                           0]
        
    # ----------------------------------------------------------
    # ----- Calculate rectangle coordinates for side view ------
    # ----------------------------------------------------------
    def rect_xy(self, width, z_center, height):
        """Rectangle coordinates centered at z_center"""
        x = [-width/2, width/2, width/2, -width/2, -width/2]
        y = [z_center - height/2, z_center - height/2,
            z_center + height/2, z_center + height/2,
            z_center - height/2]
        return x, y

    # -----------------------------------------------------------
    # --------------- Check if point is inside ------------------
    # -----------------------------------------------------------
    def in_scintillator_xy(self, x, y):
        """Check if (x, y) point in top view is inside scintillator bounds."""
        return (self.scin_xmin <= x) & (x <= self.scin_xmax) & \
            (self.scin_ymin <= y) & (y <= self.scin_ymax)
    
    # ---------------------------------------------------------------------------
    # --------------- ETA region additions --------------------------------------
    # ---------------------------------------------------------------------------
    
    # ---- ETA LAYOUT: data + helpers -------------------------------------------
    def set_eta_layout(self, y_vals, x_left, x_right):
        """
        Define eta regions in ME0 layers

        Inputs are arrays (len = Nbreaks) describing the polygon *edges* from top to bottom:
        y_vals : vertical breaks top -> bottom (m)
        x_left : left boundary at each y
        x_right: right boundary at each y
        This creates N-1 polygons (eta 1..N-1), from (y_i -> y_{i+1}).
        """

        # Convert inputs to numpy arrays and check validity
        y_vals = np.asarray(y_vals, float)
        x_left = np.asarray(x_left, float)
        x_right = np.asarray(x_right, float)
        assert y_vals.size == x_left.size == x_right.size >= 2, "Eta inputs must be same length ≥2."

        # Store input arrays
        self.eta_y = y_vals
        self.eta_xl = x_left
        self.eta_xr = x_right

        # Initialize polygons and paths
        polys = []

        # Build (N-1) polygons for each eta region
        for i in range(len(y_vals)-1):
            # Define polygon corners
            poly = np.array([
                [x_left[i],  y_vals[i]],
                [x_right[i], y_vals[i]],
                [x_right[i+1], y_vals[i+1]],
                [x_left[i+1],  y_vals[i+1]],
                [x_left[i],  y_vals[i]],    # close
            ], dtype=float)

            # Add polygon to list
            polys.append(poly)
        
        # Store polygons and paths
        self.eta_polys = polys
        self.eta_paths = [Path(p) for p in self.eta_polys]  # for fast containment

    def enable_default_me0_eta(self):
        """
        Load the 9 y-breaks & edge x-values
        Top → bottom. This defines 8 eta regions (1..8).
        """
        # Eta region coordinates in mm
        y_mm = [785, 650, 526, 415, 315, 225, 142, 67, 0]
        xl_mm = [-230,-230,-211,-192,-175,-157, -142, -129, -117]
        xr_mm = [ 230, 230, 211, 192, 175, 157,  142,  129,  117]

        # Convert to meters
        y =  np.array(y_mm , float)/1000.0
        xl = np.array(xl_mm, float)/1000.0
        xr = np.array(xr_mm, float)/1000.0

        # Set eta layout / create polygons
        self.set_eta_layout(y, xl, xr)

    def which_eta(self, x, y):
        """
        Return eta index (1..8) for points (x,y), or 0 if outside all regions.
        Vectorized over x,y arrays of same length. Returns int array.
        """

        # Check that eta layout is defined
        if not hasattr(self, "eta_paths"):
            raise RuntimeError("Eta layout not defined. Call enable_default_me0_eta() or set_eta_layout().")
        
        # Combine x,y into a single array of points
        pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
        
        #  Initialize output array with zeros (0 = outside all eta regions)
        out = np.zeros(pts.shape[0], dtype=int)

        # Iterate over all eta regions and mark points inside
        for i, path in enumerate(self.eta_paths, start=1):
            # Find points inside current eta region
            sel = path.contains_points(pts)

            # Assign eta index to those points
            out[sel] = i

        # Return array of eta indices
        return out
    
    # ---- Helpers for eta layout ------------------------------------------------

    def width_at_y(self, y):
        """Full width at vertical coordinate y (m), 0 at bottom - total_length at top."""
        y = float(np.clip(y, 0.0, self.total_length))
        y_taper_top = self.total_length - self.rect_length  # start of taper
        if y >= y_taper_top:
            return self.top_width
        frac = y / y_taper_top if y_taper_top > 0 else 0.0
        return self.bottom_width + (self.top_width - self.bottom_width) * frac

    def eta_breaks_from_mm(self, y_mm, mm_bottom=None, mm_top=None):
        """
        Convert a list of mm positions (bottom..top or top..bottom) into meters
        on [0, total_length], returned ASCENDING (bottom->top).
        """
        y_mm = np.asarray(y_mm, dtype=float)
        if mm_bottom is None:
            mm_bottom = float(np.min(y_mm))
        if mm_top is None:
            mm_top = float(np.max(y_mm))
        if mm_top <= mm_bottom:
            raise ValueError("mm_top must be > mm_bottom.")
        # normalize to [0,1], then scale to [0,total_length]
        y_norm = (y_mm - mm_bottom) / (mm_top - mm_bottom)
        y_m = y_norm * self.total_length
        y_m = np.sort(y_m)  # ensure ascending
        return y_m

    def eta_polygons(self, y_breaks_m):
        """
        Build trapezoidal polygons for each eta band defined by y_breaks in meters.
        Returns: list of (4,2) arrays of corners [-w/2,y]..[+w/2,y] (clockwise).
        """

        # Convert to numpy array of floats
        yb = np.asarray(y_breaks_m, dtype=float)

        # Check validity
        if not (np.all(np.diff(yb) >= 0) and yb.size >= 2):
            raise ValueError("y_breaks must be ascending with at least two values.")
        
        # Build polygons
        polys = []
        for y0, y1 in zip(yb[:-1], yb[1:]):
            w0 = self.width_at_y(y0)
            w1 = self.width_at_y(y1)
            polys.append(np.array([
                [-w1/2, y1],
                [ +w1/2, y1],
                [ +w0/2, y0],
                [ -w0/2, y0]
            ]))
        return polys