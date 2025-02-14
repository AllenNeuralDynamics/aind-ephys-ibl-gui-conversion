import numpy as np
import os

import zarr


import ants
import json

from aind_mri_utils.measurement import find_line_eig
from pathlib import Path
from aind_mri_utils.file_io.neuroglancer import read_neuroglancer_annotation_layers
from iblatlas import atlas

def create_slicer_fcsv(filename,pts_mat,direction = 'LPS',pt_orientation = [0,0,0,1],pt_visibility = 1,pt_selected = 0, pt_locked = 1):
    """
    Save fCSV file that is slicer readable.
    """
    # Create output file
    OutObj = open(filename,"w+")
    
    header0 = '# Markups fiducial file version = 4.11\n'
    header1 = '# CoordinateSystem = '+ direction+'\n'
    header2 = '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n'
    
    OutObj.writelines([header0,header1,header2])
    
    outlines = []
    for ii in range(pts_mat.shape[0]):
        outlines.append(
            str(ii+1) +','+ 
            str(pts_mat[ii,0])+','+ 
            str(pts_mat[ii,1])+','+ 
            str(pts_mat[ii,2])+
            f',{pt_orientation[0]},{pt_orientation[1]},{pt_orientation[2]},{pt_orientation[3]},'+
            f'{pt_visibility},{pt_selected},{pt_locked},'+ 
            str(ii)+',,vtkMRMLScalarVolumeNode1\n')
    
    OutObj.writelines(outlines)
    OutObj.close()
    
def probe_df_to_fcsv(probe_data,extrema,results_folder,offset=(0,0,0)):
    """
    Convert probe data to fcsv files for slicer
    """
    unq = np.unique(probe_data.tree_id)
    probes = {}
    for ii,uu in enumerate(unq):
        this_probe_data = probe_data[probe_data.tree_id==uu]
        x = extrema[0]-(this_probe_data.x/1000).values+offset[0]
        y = (this_probe_data.y/1000).values+offset[1]
        z = -(this_probe_data.z/1000).values+offset[2]    
        probes[str(uu)] = np.vstack([x,y,z]).T
        create_slicer_fcsv(os.path.join(results_folder,f'test{uu}.fcsv'),probes[str(uu)],direction = 'LPS')
        
    return probes

def projected_onto_line(points,line_N,line_P):
    """
    Projects a set of points onto a line defined by a direction vector and a point on the line.

    Args:
        points (np.ndarray): An array of points to be projected. Shape (n_points, n_dimensions).
        line_N (np.ndarray): A direction vector of the line. Shape (n_dimensions,).
        line_P (np.ndarray): A point on the line. Shape (n_dimensions,).

    Returns:
        np.ndarray: An array of projected points. Shape (n_points, n_dimensions).

    Example:
        points = np.array([[1, 2, 3], [4, 5, 6]])
        line_N = np.array([1, 0, 0])
        line_P = np.array([0, 0, 0])
        projected_points = projected_onto_line(points, line_N, line_P)

    Note:
        This function normalizes the direction vector `line_N` before performing the projection.

    This docstring was generated by ChatGPT.
    """
    line_N = line_N/np.linalg.norm(line_N)
    projL = (points-line_P)@line_N
    return line_P+np.outer(projL,line_N)

def order_annotation_pts(points,axis = 2,order = 'desending'):
    N,pt = find_line_eig(points)
    proj = projected_onto_line(points,N,pt)
    this_order = np.argsort(proj[:,2])
    if order == 'desending':
        this_order = this_order[::-1]
    return points[this_order,:]


def read_json_as_dict(filepath: str):
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary



def __read_zarr_image(image_path):
    """
    Reads a zarr image

    Parameters
    -------------
    image_path: PathLike
        Path where the zarr image is located

    Returns
    -------------
    np.array
        Numpy array with the zarr image
    """
    image_path = str(image_path)
    zarr_img = zarr.open(image_path, mode="r")
    img_array = np.asarray(zarr_img)
    img_array = np.squeeze(img_array)
    return img_array

def check_orientation(img: np.array, params: dict, orientations: dict):
    """
    Checks aquisition orientation an makes sure it is aligned to the CCF. The
    CCF orientation is:
        - superior_to_inferior
        - left_to_right
        - anterior_to_posterior

    Parameters
    ----------
    img : np.array
        The raw image in its aquired orientatin
    params : dict
        The orientation information from processing_manifest.json
    orientations: dict
        The axis order of the CCF reference atals

    Returns
    -------
    img_out : np.array
        The raw image oriented to the CCF
    """

    orient_mat = np.zeros((3, 3))
    acronym = ["", "", ""]

    for k, vals in enumerate(params):
        direction = vals["direction"].lower()
        dim = vals["dimension"]
        if direction in orientations.keys():
            ref_axis = orientations[direction]
            orient_mat[dim, ref_axis] = 1
            acronym[dim] = direction[0]
        else:
            direction_flip = "_".join(direction.split("_")[::-1])
            ref_axis = orientations[direction_flip]
            orient_mat[dim, ref_axis] = -1
            acronym[dim] = direction[0]

    # check because there was a bug that allowed for invalid spl orientation
    # all vals should be postitive so just taking absolute value of matrix
    if "".join(acronym) == "spl":
        orient_mat = abs(orient_mat)

    original, swapped = np.where(orient_mat)
    img_out = np.moveaxis(img, original, swapped)

    out_mat = orient_mat[:, swapped]
    for c, row in enumerate(orient_mat.T):
        val = np.where(row)[0][0]
        if row[val] == -1:
            img_out = np.flip(img_out, c)
            out_mat[val, val] *= -1

    return img_out, orient_mat, out_mat


def get_highest_level_info(filepath,return_order = 'xyz'):


    with open(os.path.join(filepath,'.zattrs')) as f:
        metadata = json.load(f)

    zarr_axis_order = [field['name'] for field in metadata['multiscales'][0]['axes']]
    highest_level = metadata['multiscales'][0]['datasets'][-1]
    scale = highest_level['coordinateTransformations'][0]['scale']
    level_path = highest_level['path']

    # Put in xyz order
    sort_scale = []
    for ii in list(return_order):
        sort_scale.append(scale[zarr_axis_order.index(ii)])

    return level_path,np.array(sort_scale)


def get_additional_channel_image_at_highest_level(image_path,
                                ants_template,
                                input_orientations,
                                template_orientations = {
                                    "anterior_to_posterior": 1,
                                    "superior_to_inferior": 2,
                                    "right_to_left": 0,
                                },scale_factor = 1e-3):
    highest_level,scale = get_highest_level_info(image_path)
    img_array = __read_zarr_image(os.path.join(image_path,highest_level))
    img_array = img_array.astype(np.double)
    img_out, in_mat, out_mat = check_orientation(
        img_array,
        input_orientations,
        template_orientations,
        )

    ants_img = ants.from_numpy(img_out, spacing=list(scale*scale_factor))
    ants_img.set_direction(ants_template.direction)
    ants_img.set_origin(ants_template.origin)
    return ants_img

def save_image_volumes(zarr_read: ants.ANTsImage, ccf_25: ants.ANTsImage, ccf_annotation_25: ants.ANTsImage, 
                       moved_image_folder: str, image_histology_results: str, manifest_df: pd.DataFrame) -> None:
    """
    Saves the volumes in image space and ccf space 
    """
    ccf_in_image_space = ants.apply_transforms(zarr_read,
            ccf_25,
            [os.path.join(moved_image_folder,'ls_to_template_SyN_0GenericAffine.mat'),
                os.path.join(moved_image_folder,'ls_to_template_SyN_1InverseWarp.nii.gz'),
                '/data/spim_template_to_ccf/syn_0GenericAffine.mat',
                '/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz',],
            whichtoinvert=[True,False,True,False],)
    ants.image_write(ccf_in_image_space,os.path.join(image_histology_results,f'ccf_in_{manifest_df.mouseid[0]}.nrrd'))

    ccf_labels_in_image_space = ants.apply_transforms(zarr_read,
                                        ccf_annotation_25,
                                        [os.path.join(moved_image_folder,'ls_to_template_SyN_0GenericAffine.mat'),
                                            os.path.join(moved_image_folder,'ls_to_template_SyN_1InverseWarp.nii.gz'),
                                            '/data/spim_template_to_ccf/syn_0GenericAffine.mat',
                                            '/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz',],
                                        whichtoinvert=[True,False,True,False],
                                        interpolator='genericLabel')
    ants.image_write(ccf_labels_in_image_space,os.path.join(image_histology_results,f'labels_in_{manifest_df.mouseid[0]}.nrrd'))

def save_annotation_outputs(annotation_file_path: Path, row: pd.Series, extrema: list, offset: list, spim_results: str,
                       moved_image_folder: str, template_results: str, ccf_results: str, brain_atlas: atlas.AllenAtlas,
                       bregma_results: str, results_folder: Path, shank_id: int = 0) -> None:
    """
    Generates the slicer files and xyz json files in image space and ccf space needed by the IBL Gui
    """ 
    probe_data = read_neuroglancer_annotation_layers(annotation_file_path, layer_names = [row.probe_id])
    this_probe_data = pd.DataFrame({'x':probe_data[row.probe_id][:,0],
                                    'y':probe_data[row.probe_id][:,1],
                                    'z':probe_data[row.probe_id][:,2]})
    x = extrema[0]-this_probe_data.x.values*1e3+offset[0]
    y  = this_probe_data.y.values*1e3+offset[1]
    z = -this_probe_data.z.values*1e3+offset[2]

    this_probe = np.vstack([x,y,z]).T
    this_probe = order_annotation_pts(this_probe)
    create_slicer_fcsv(os.path.join(spim_results,f'{row.probe_id}.fcsv'),this_probe,direction = 'LPS')

    # Move probe into template space.
    this_probe_df = pd.DataFrame({'x':this_probe[:,0],'y':this_probe[:,1],'z':this_probe[:,2]})
    # Transform into template space
    this_probe_template = ants.apply_transforms_to_points(3,this_probe_df,[os.path.join(moved_image_folder,'ls_to_template_SyN_0GenericAffine.mat'),
                                                                            os.path.join(moved_image_folder,'ls_to_template_SyN_1InverseWarp.nii.gz')],
                                                                        whichtoinvert=[True,False])
    create_slicer_fcsv(os.path.join(template_results,f'{row.probe_id}.fcsv'),this_probe_template.values,direction = 'LPS')

    # Move probe into ccf space
    this_probe_ccf = ants.apply_transforms_to_points(3,this_probe_template,['/data/spim_template_to_ccf/syn_0GenericAffine.mat',
                                                    '/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz'],
                                    whichtoinvert=[True,False])
    create_slicer_fcsv(os.path.join(ccf_results,f'{row.probe_id}.fcsv'),this_probe_ccf.values,direction = 'LPS')

    # Transform into ibl x-y-z-picks space
    ccf_mlapdv = this_probe_ccf.values.copy()*1000
    ccf_mlapdv[:,0] = -ccf_mlapdv[:,0]
    ccf_mlapdv[:,1] = ccf_mlapdv[:,1]
    ccf_mlapdv[:,2] = -ccf_mlapdv[:,2]
    bregma_mlapdv = brain_atlas.ccf2xyz(ccf_mlapdv, ccf_order='mlapdv')*1000000

    xyz_image_space = this_probe_data[['x', 'y', 'z']].to_numpy()
    xyz_image_space[:, 0] = (extrema[0] - (xyz_image_space[:, 0] * 1000)) * 1000
    xyz_image_space[:, 1] = xyz_image_space[:, 1] * 1000000
    xyz_image_space[:, 2] = xyz_image_space[:, 2] * 1000000

    xyz_picks_image_space = {'xyz_picks':xyz_image_space.tolist()}
    xyz_picks_ccf = {'xyz_picks': bregma_mlapdv.tolist()}

    # Save this in two locations. First, save sorted by filename
    with open(os.path.join(bregma_results,f'{row.probe_id}_image_space.json'), "w") as f:
        # Serialize data to JSON format and write to file
        json.dump(xyz_picks_image_space, f)

    with open(os.path.join(bregma_results,f'{row.probe_id}_ccf.json'), "w") as f:
        # Serialize data to JSON format and write to file
        json.dump(xyz_picks_ccf, f)

    # Second, save the XYZ picks to the sorting folder for the gui.
    # This step will be skipped if there was a problem with the ephys pipeline.
    if os.path.isdir(os.path.join(results_folder,str(row.probe_name))):
        if shank_id > 0:
            image_space_filename = f'xyz_picks_shank{shank_id}_image_space.json'
            ccf_space_filename = f'xyz_picks_shank{shank_id}.json'
        else:
            image_space_filename = 'xyz_picks_image_space.json'
            ccf_space_filename = 'xyz_picks.json'

        with open(os.path.join(results_folder,str(row.probe_name), image_space_filename),"w") as f:
            json.dump(xyz_picks_image_space, f)
        
        with open(os.path.join(results_folder,str(row.probe_name), ccf_space_filename),"w") as f:
            json.dump(xyz_picks_ccf, f)