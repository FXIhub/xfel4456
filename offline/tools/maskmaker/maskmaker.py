# mask and data are raveled

import argparse
import h5py
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import scipy
import signal
import os
import pickle

import time

class Application:
    def __init__(self, data, ij_map, xyz_map, background_mask, mask, d, Z):
        self.Z = Z
        self.frame_index = 0
         
        self.data = data[self.frame_index].ravel()

        # corrected jungfrau data has wierd values
        self.data[~np.isfinite(self.data)] = 0
        self.data_max = self.data.max()
        
        self.z_data = data

        self.xmin, self.ymin, self.dx = d
        
        self.pixel_map       = ij_map
        self.background_mask = background_mask
        self.mask            = mask
        self.data_shape      = data[self.frame_index].shape
        self.image_shape     = background_mask.shape
        self.xy_map          = xyz_map
        
        self.trans_mask = np.ones(self.image_shape, dtype=bool)
         
        self.trans = np.zeros(self.image_shape, dtype=data.dtype)
    
        self.trans[self.pixel_map] = self.data

        # store the flattened data indices in 2D image space
        self.index_map                 = -np.ones(self.image_shape, dtype=int)
        self.index_map[self.pixel_map] = np.arange(self.data.size)
        
        self.display_RGB = np.zeros(self.image_shape + (3,), dtype = self.data.dtype)
        
        self.mask_edges  = False
        self.panel_edges = edges(self.data_shape)

        self.in_replot = False
        
        self.initUI()
        
    def initUI(self):
        signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
        
        # Always start by initializing Qt (only once per application)
        self.app = QtWidgets.QApplication([])
        
        # Define a top-level widget to hold everything
        w = QtWidgets.QWidget()
        
        # 2D plot for the cspad and mask
        self.plot = pg.ImageView()

        # add a + at the origin
        # x=0, i = -xmin / dx + 0.5
        i0 = -self.xmin / self.dx + 0.5
        j0 = -self.ymin / self.dx + 0.5
        scatter = pg.ScatterPlotItem([{'pos': (i0, j0), 'size': 5, 'pen': pg.mkPen('r'), 'brush': pg.mkBrush('r'), 'symbol': '+'}])
        self.plot.addItem(scatter)
        
        # save mask button
        save_button = QtWidgets.QPushButton('save mask')
        save_button.clicked.connect(self.save_mask)
        
        # rectangular ROI selection
        self.roi = pg.RectROI([-200,-200], [100, 100])
        self.plot.addItem(self.roi)
        self.roi.setZValue(10)                       # make sure ROI is drawn above image
        ROI_button = QtWidgets.QPushButton('mask rectangular ROI')
        ROI_button.clicked.connect(lambda : self.mask_ROI(self.roi))
         
        # circular ROI selection
        self.roi_circle = pg.CircleROI([-200,200], [101, 101])
        self.plot.addItem(self.roi_circle)
        self.roi.setZValue(10)                       # make sure ROI is drawn above image
        ROI_circle_button = QtWidgets.QPushButton('mask circular ROI')
        ROI_circle_button.clicked.connect(lambda : self.mask_ROI_circle(self.roi_circle))
        
        # histogram mask button
        hist_button = QtWidgets.QPushButton('mask outside histogram')
        hist_button.clicked.connect(self.mask_hist)
        
        # dilate button
        dilate_button = QtWidgets.QPushButton('dilate mask')
        dilate_button.clicked.connect(self.dilate_mask)

        # errode button
        errode_button = QtWidgets.QPushButton('errode mask')
        errode_button.clicked.connect(self.errode_mask)
        
        # Brush
        self.brush_img = None
        self.brush_button = QtWidgets.QPushButton('brush')
        self.brush_button.clicked.connect(self.use_brush)
        self.brush_button.setCheckable(True)

        self.brush_size = QtWidgets.QSpinBox(value=10, minimum=1)
        self.brush_size.valueChanged.connect(self.change_brush)
        
        add_button = QtWidgets.QPushButton('add')
        add_button.clicked.connect(self.add_brush)
            
        discard_button = QtWidgets.QPushButton('discard')
        discard_button.clicked.connect(self.discard_brush)
        
        # toggle / mask / unmask checkboxes
        self.toggle_checkbox   = QtWidgets.QCheckBox('toggle')
        self.mask_checkbox     = QtWidgets.QCheckBox('mask')
        self.unmask_checkbox   = QtWidgets.QCheckBox('unmask')
        self.mask_checkbox.setChecked(True)   
        
        toggle_group           = QtWidgets.QButtonGroup()#"masking behaviour")
        toggle_group.addButton(self.toggle_checkbox)   
        toggle_group.addButton(self.mask_checkbox)   
        toggle_group.addButton(self.unmask_checkbox)   
        toggle_group.setExclusive(True)
        
        # asic edges button
        edges_button = QtWidgets.QPushButton('panel edges')
        edges_button.clicked.connect( self.mask_edge_pixels )
        
        # mouse hover ij value label
        ij_label = QtWidgets.QLabel()
        disp = 'ss fs {0:5} {1:5}   value {2:2}'.format('-', '-', '-')
        ij_label.setText(disp)
        self.plot.scene.sigMouseMoved.connect( lambda pos: self.mouseMoved(ij_label, pos) )
        
        # asic edges checkbox
        edges_checkbox = QtWidgets.QCheckBox('asic edges')
        edges_checkbox.stateChanged.connect( self.update_mask_edges )
        
        # mouse click mask 
        self.plot.scene.sigMouseClicked.connect( lambda click: self.mouseClicked(self.plot, click) )
        
        if self.Z > 1 :
            # add a z-slider for image selection
            z_sliderW = pg.PlotWidget()
            z_sliderW.plot(np.arange(self.Z), pen=(255, 150, 150))
            z_sliderW.setFixedHeight(50)
            
            # vline
            self.vline = z_sliderW.addLine(x = 0, movable=True, bounds = [0, self.Z-1])
                
            self.vline.sigPositionChanged.connect(self.replot_frame)
        
        ## Add widgets to the layout in their proper positions
        # stack up sidepanel widgets
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(save_button)
        vbox.addWidget(ROI_button)
        vbox.addWidget(ROI_circle_button)
        vbox.addWidget(hist_button)
        vbox.addWidget(dilate_button)
        vbox.addWidget(errode_button)
        vbox.addWidget(edges_button)
        
        vbox.addWidget(self.toggle_checkbox)
        vbox.addWidget(self.mask_checkbox)
        vbox.addWidget(self.unmask_checkbox)

        brush_layout = QtWidgets.QGridLayout()
        brush_layout.addWidget(self.brush_button, 0, 0)
        brush_layout.addWidget(self.brush_size, 0, 1)
        brush_layout.addWidget(add_button, 1, 0)
        brush_layout.addWidget(discard_button, 1, 1)
        
        vbox.addLayout(brush_layout)
        
        vbox.addStretch(1)
        vbox.addWidget(edges_checkbox)
        
        # Create a grid layout to manage the widgets size and position
        layout = QtWidgets.QGridLayout()
        w.setLayout(layout)
        
        layout.addLayout(vbox, 0, 0)
        layout.addWidget(self.plot, 0, 1)
        if self.Z > 1 :
            layout.addWidget(z_sliderW, 1, 0, 1, 2)
            layout.addWidget(ij_label,  2, 0, 1, 2)
        else :
            layout.addWidget(ij_label,  1, 0, 1, 2)
        
        layout.setColumnStretch(1, 1)
        
        # display the image
        self.updateDisplayRGB(auto = True)
        
        # centre the circle initially 
        r =  self.roi_circle.size()[0]/2.
        self.roi_circle.setPos([i0-r, j0-r])
        
        ## Display the widget as a new window
        w.resize(800, 480)
        w.show()
        
        ## Start the Qt event loop
        self.app.exec_()

    def replot_frame(self):
        if self.in_replot:
            return
        try:
            self.in_replot = True
            i = int(self.vline.value())
            if self.frame_index != i :
                self.frame_index = i
                self.data[:] = self.z_data[i].ravel()
                self.data[~np.isfinite(self.data)] = 0
                self.data_max = self.data.max()
                self.trans[~self.background_mask] = self.data[self.pixel_map]
                self.updateDisplayRGB()
        finally:
            self.in_replot = False

    def save_mask(self):
        print('saving good pixels to mask.h5/entry_1/good_pixels')
        with h5py.File('mask.h5', 'w') as f:
            f['entry_1/good_pixels'] = self.mask.reshape(self.data_shape)
            f['entry_1/bad_pixels'] = ~self.mask.reshape(self.data_shape)
        print('Done!')
    
    def mask_ROI(self, roi):
        # top left of roi in physical units
        x = roi.pos()[0] * self.dx + self.xmin
        y = roi.pos()[1] * self.dx + self.ymin
        
        # width and height of roi in physical units
        dx = roi.size()[0] * self.dx
        dy = roi.size()[1] * self.dx
        
        m = (self.xy_map[0] > x) * (self.xy_map[0] < (x+dx)) * (self.xy_map[1] > y) * (self.xy_map[1] < (y+dy))
        
        self.update_mask(m.ravel())
    
    def mask_ROI_circle(self, roi):
        # get the xy coords of the centre and the radius
        r = dx * roi.size()[0]/2.
        x = roi.pos()[0] * self.dx + self.xmin + r
        y = roi.pos()[1] * self.dx + self.ymin + r
        
        r_map = ((self.xy_map[0]-x)**2 + (self.xy_map[1]-y)**2)**0.5
        self.update_mask((r_map < r).ravel())
    
    def update_mask(self, mask):
        if self.toggle_checkbox.isChecked():
            self.mask[mask] = ~self.mask[mask]
        elif self.mask_checkbox.isChecked():
            self.mask[mask] = False
        elif self.unmask_checkbox.isChecked():
            self.mask[mask] = True
        self.updateDisplayRGB()

    def updateDisplayRGB(self, auto = False):
        """
        Make an RGB image (N, M, 3) (pyqt will interprate this as RGB automatically)
        with masked pixels shown in blue at the maximum value of the cspad. 
        This ensures that the masked pixels are shown at full brightness.
        """
        # now to map data to 2D image we have:
        # im[~background] = data.ravel()[indices_image_space]
        self.trans_mask[self.pixel_map] = self.mask
        
        # convert to RGB
        # Set masked pixels to B
        self.display_RGB[:, :, 0] = (self.trans * self.trans_mask).reshape(self.image_shape)
        self.display_RGB[:, :, 1] = (self.trans * self.trans_mask).reshape(self.image_shape)
        self.display_RGB[:, :, 2] = (self.trans + (self.data_max - self.trans) * ~self.trans_mask).reshape(self.image_shape)
        
        print('data max', self.frame_index, self.data_max)
        
        if auto :
            self.plot.setImage(self.display_RGB)
        else :
            self.plot.setImage(self.display_RGB, autoRange = False, autoLevels = False, autoHistogramRange = False)

    def mask_hist(self):
        min_max = self.plot.getHistogramWidget().item.getLevels()

        m = (self.data < min_max[0]) + (self.data > min_max[1])
        self.update_mask(m.ravel())

    def dilate_mask(self):
        """
        do this on a per-panel basis
        """
        
        # loop over panels (last 2 dimensions in this case)
        m = ~self.mask.copy().reshape((-1,) + self.data_shape[-2:])
        for k in range(m.shape[0]):
            m[k] = binary_dilation(m[k])
        
        # ignore mask/unmask/toggle modes (too confusing)
        self.mask[:] = ~m.ravel()
        self.updateDisplayRGB()
        
    def errode_mask(self, mask = None):
        # loop over panels (last 2 dimensions in this case)
        m = ~self.mask.copy().reshape((-1,) + self.data_shape[-2:])
        for k in range(m.shape[0]):
            m[k] = binary_erosion(m[k], border_value=1)
        
        # ignore mask/unmask/toggle modes (too confusing)
        self.mask[:] = ~m.ravel()
        self.updateDisplayRGB()

    def generate_brush_kernel(self):
        size = self.brush_size.value()
        r = size/2.
        cx, cy = (size - 1)/2., (size - 1)/2.
        x, y = np.ogrid[-cx:size-cx, -cy:size-cy]
        kernel = np.zeros((size, size, 4))
        kernel[:,:,0][x*x + y*y < r*r] = 1
        kernel[:,:,3][x*x + y*y < r*r] = 1
        return kernel

    def use_brush(self):
        if self.brush_button.isChecked():
            self.app.setOverrideCursor(QtCore.Qt.CrossCursor)
            img = self.plot.getImageItem()
            self.brush_img = pg.ImageItem(np.zeros((img.image.shape[0], img.image.shape[1], 4)))
            self.plot.addItem(self.brush_img)
            kernel = self.generate_brush_kernel()
            self.brush_img.setLevels([0, 1])
            self.brush_img.setDrawKernel(kernel, mask=kernel, center=(kernel.shape[0]//2, kernel.shape[1]//2), mode='set')
        elif self.brush_img:
            self.discard_brush()

    def change_brush(self):
        if self.brush_img:
            kernel = self.generate_brush_kernel()
            self.brush_img.setLevels([0, 1])
            self.brush_img.setDrawKernel(kernel, mask=kernel, center=(kernel.shape[0]//2, kernel.shape[1]//2), mode='set')
        else:
            pass

    def discard_brush(self):
        if self.brush_button.isChecked():
            self.app.restoreOverrideCursor()
            self.plot.removeItem(self.brush_img)
            self.brush_img.clear()
            self.brush_img = None
            self.brush_button.toggle()
    
    def add_brush(self):
        if not self.brush_button.isChecked():
            return
        
        m = np.zeros_like(self.mask)
        for i0, j0 in np.transpose(np.where(self.brush_img.image[:,:,0] > 0)):
            if (0 <= i0 < self.image_shape[0]) and (0 <= j0 < self.image_shape[1]) and not self.background_mask[i0*self.image_shape[1]+ j0]:
                m[self.index_map[i0, j0]] = True
                    
        self.discard_brush()
        self.update_mask(m)
        
    def mask_edge_pixels(self):
        self.update_mask(~self.panel_edges.ravel())

    def update_mask_edges(self, state):
        amask = asic_edges(self.data_shape).ravel()
        if state > 0 :
            print('adding asic edges to the mask')
            self.mask[~amask] = False
        else :
            print('removing asic edges from the mask')
            self.mask[~amask] = True
        self.updateDisplayRGB()

    def mouseMoved(self, ij_label, pos):
        img = self.plot.getImageItem()
        i = int(round(img.mapFromScene(pos).x()-0.5))
        j = int(round(img.mapFromScene(pos).y()-0.5))
        x = i * self.dx + self.xmin
        y = j * self.dx + self.ymin
        
        v = -1
        value = 0
        
        if (0 <= i < self.image_shape[0]) and (0 <= j < self.image_shape[1]) :
            k = self.index_map[i, j]
            if k != -1 :
                v = np.unravel_index(k, self.data_shape)
                value = self.trans[i, j]
        
        ij_label.setText('data index {} value {:5.2e} x,y {:5.2e}, {:5.2e}'.format(v, value, x, y))

    def mouseClicked(self, plot, click):
        if self.brush_button.isChecked():
            return
        
        if click.button() == 1:
            img = plot.getImageItem()
            
            i = int(round(img.mapFromScene(click.pos()).x()-0.5))
            j = int(round(img.mapFromScene(click.pos()).y()-0.5))
            if (0 <= i < self.image_shape[0]) and (0 <= j < self.image_shape[1]) :
                k = self.index_map[i, j]
                if k != -1 :
                    if self.toggle_checkbox.isChecked():
                        self.mask[k]         = ~self.mask[k]
                    elif self.mask_checkbox.isChecked():
                        self.mask[k]         = False
                    elif self.unmask_checkbox.isChecked():
                        self.mask[k]         = True

                    self.updateDisplayRGB()
    

#shape = (16, 128, 512)
def asic_edges(shape):
    # loop over 2D slabs (edge == False is edge)
    edges = np.ones(shape, dtype=bool)
    # bottom fs edge
    edges[..., ::128] = False
    edges[..., 127::128] = False

    edges[..., ::128, :] = False
    edges[..., 127::128, :] = False
    
    return edges

def edges(shape):
    # loop over 2D slabs (edge == False is edge)
    edges = np.ones(shape, dtype=bool)
    # bottom fs edge
    edges[..., 0] = False
    # top fs edge
    edges[..., -1] = False
        
    # bottom ss edge
    edges[..., 0, :] = False
    # top fs edge
    edges[..., -1, :] = False
    return edges



def parse_cmdline_args():
    parser = argparse.ArgumentParser(description='maskmaker - mask making, but with a mouse!')
    parser.add_argument('data', type=str, help="filename for the hdf5 image file. specify the filename and datapath as e.g. ../../../tests/powder_r0040.h5/data")
    parser.add_argument('-g', '--geometry', type=str, help="path to the xyz map for the image e.g. jungfrau_xyz_map.pickle")
    parser.add_argument('-m', '--mask', type=str, help="path to the h5file of the starting mask")
    return parser.parse_args()

# if fnam is of the type "/loc/filename.h5/dataset"
def get_fnam_and_path(fnam):
    if len(fnam.split('.h5/')) > 1 :
        dataset = fnam.split('.h5/')[1]
        fnam = fnam.split('.h5/')[0] + '.h5'
    elif len(fnam.split('.cxi/')) > 1 :
        dataset = fnam.split('.cxi/')[1]
        fnam = fnam.split('.cxi/')[0] + '.cxi'
    else :
        dataset = '/'
    return fnam, dataset

def generate_pixel_lookup(xyz):
    # choose xy bounds
    xmin = xyz[0].min()
    xmax = xyz[0].max()
    
    ymin = xyz[1].min()
    ymax = xyz[1].max()
    
    # choose sampling
    dx = 177e-6 / 2
    
    shape = (int( (xmax-xmin)/dx ) + 2, int( (ymax-ymin)/dx ) + 2)

    # pixel coordinates in im
    ss = np.round((xyz[0] - xmin) / dx).astype(int)
    fs = np.round((xyz[1] - ymin) / dx).astype(int)
    
    i, j = np.indices(shape)
    xy_map = np.empty((2,) + shape, dtype=float)
    xy_map[0] = dx * i
    xy_map[1] = dx * j
    
    # now use pixel indices as labels
    i = np.arange(xyz[0].size)
     
    # create an image of the data raveled indices
    im = -np.ones(shape, dtype=int)
    im[ss.ravel(), fs.ravel()] = i
    
    # label image
    # problem, the labells dont equal i
    #l = skimage.measure.label(im, background=-1)
    
    # expand by oversampling rate (to fill gaps)
    l = skimage.segmentation.expand_labels(im+1, distance = 2)
        
    # set background mask
    background_mask = (l.ravel()==0).copy()
    
    # now subtract 1 from labels to turn them into pixel indices
    l -= 1
    
    indices_image_space = l.ravel()[~background_mask].copy()
    
    # now to map data to 2D image we have:
    # im[~background] = data.ravel()[indices_image_space]
    return indices_image_space, background_mask, shape, (xmin, ymin, dx), im, i, l

    
if __name__ == '__main__':
    args = parse_cmdline_args()
    
    args.data, args.data_path = get_fnam_and_path(args.data)
    with h5py.File(args.data) as f:
        data = f[args.data_path][()]
    
    if args.geometry is not None :
        args.geometry, args.geometry_path = get_fnam_and_path(args.geometry)
        with h5py.File(args.geometry) as f:
            xyz = f[args.geometry_path][()]
    else :
        fnam_xyz = '/gpfs/exfel/exp/SPB/202302/p004456/scratch/det/jungfrau_xyz_map.pickle'
        xyz = pickle.load(open(fnam_xyz, 'rb'))
    
    if args.mask is not None :
        args.mask, args.mask_path = get_fnam_and_path(args.mask)
        with h5py.File(args.mask) as f:
            mask = f[args.mask_path][()]
    else :
        mask = np.ones(xyz.shape[1:], dtype=bool)

    # if data has a larger dimension than the xyz values then add slider
    if len(data.shape) == len(xyz.shape[1:]) :
        Z = 1
        # add a z-index to data
        data = data[None, ...]
    elif len(data.shape) == (1+len(xyz.shape[1:])) :
        # check if z-index (frame number) is at 0:
        for dim in range(len(data.shape)) :
            if data.shape[dim] not in xyz.shape[1:] and dim > 0 :
                data = np.swapaxes(data, dim, 0).copy()
                print(f'swapping axis {dim} and 0 in data. {data.shape}')
        Z = data.shape[0]
    else :
        print('cannot interpret data shape')
    
    # generate ND -> 2D lookup table
    #indices_image_space, background_mask, im_shape, (xmin, ymin, dx), im, i, l = generate_pixel_lookup(xyz)
    
    # image[index_map] = data.ravel()
    
    # convert to pixel units
    dx = 75e-6 # pixel size
    x = xyz[0] / dx
    y = xyz[1] / dx
    
    # find the smallset size of the 2D that contains all xy values but is symmetric about the origin
    N = 2 * int(np.max(np.abs(x))) + 2
    M = 2 * int(np.max(np.abs(y))) + 2
    image_shape = (N, M)
    
    # convert to pixel index values
    i = np.array(x, dtype=int) + N//2 - 1
    j = np.array(y, dtype=int) + M//2 - 1

    # get the minimum x and y value in the 2D map
    # seems harder than it should be
    # find a single x/y value for a pixel then extrapolate
    txy = (xyz[0].ravel()[0], xyz[1].ravel()[0])
    tij = (i.ravel()[0], j.ravel()[0])
    
    xmin = txy[0] - dx * tij[0]
    ymin = txy[1] - dx * tij[1]
    
    # the index map points every data pixel to an i,j value
    ij_map = (i.ravel(), j.ravel())
    
    # now create a background mask in image space
    background_mask = np.ones(image_shape, dtype=bool)
    background_mask[ij_map] = False
    
    # test: 
    #data = (xyz[0]**2 + xyz[1]**2)**.5
    #data = xyz[1]

    # start the gui
    Application(data, ij_map, xyz, background_mask, mask.ravel().copy(), (xmin, ymin, dx), Z)
