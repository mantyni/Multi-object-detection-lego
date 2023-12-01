"""
This is the current latest version of the script used in both Blender 2.93 & 3.65 scene files. 
This script generates randomly placed Lego part on a varied background renderings using Blender object and camera manipulation API. 
The goal of the script was to create a dataset to train Machine Learning object detection models.

Some parts of the code were inspired and borrowed from forums in the Blender community.

If the code does not work please create a github issue and I will look into as soon as I can.
"""

import os
import math
import mathutils
import random
from datetime import datetime

import bpy
from pascal_voc_writer import Writer


# Choose one of rendering option as run_type
#1. Render all
#2. Render batch
#3. Render individual parts
#4. Render backgrounds

# Run once with run_type = 1. This will generate 100 images.
# Afterwards run script multiple times with run_type = 2. This will generate 500 extra images. 
# Suggested 4-5 runs on run_type 2 to generate ~2,5k images for dataset with 6 lego parts.
# First run will set batch 100 and next runs can be 500. 
# For me Blender crashes if more 400 images @ 640x640 resolution generated in one script run.

run_type = 2

num = 255 
colors = []
materials = []
background = []
objects = []
object_count = []
start_range = 0 
step_size = 15 # Default 15. Choose step size for part rotation when rendering indivdual parts. 
batch_size = 100 # How many images to render in first run on run_type_1. For me, more than 
i = 0

# Define colors, based on offical Lego RGB colors: 
# http://www.jennyscrayoncollection.com/2021/06/all-current-lego-colors.html
colors0=[(221, 26, 33), # Red 
        (0, 146, 71), # Green  
        (0,108,183), # Blue 
        (255,205,3), # Yellow
        (232, 80, 156), # Cyan
        (160,161,159)] # Magenta

# Set output resolution
r_settings = bpy.context.scene.render
r_settings.resolution_x = 300
r_settings.resolution_y = 300


print("######################")
print("Rendering lego parts. ", datetime.now())
print("Run_type: ", run_type)
print()

# Define output directory for images and annotations
directory = 'lego_renders'
if not os.path.exists(directory):
    print("Creating directory for output")
    os.makedirs(directory+'/images')
    os.makedirs(directory+'/annotations')

output_dir = os.getcwd() + '/' + directory
output_dir_images = output_dir + '/images'
output_dir_annotations = output_dir + '/annotations'

print("Rendering output directory: ", output_dir)

# When generating renderings in batches on run_type_2 select such size that does not crash Blender.
# 500 for me worked well.
if run_type == 2:
    batch_size = 500 

# FUNCTIONS:

# Update camera location:
def update_camera(camera, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), distance=10.0):
    """
    Focus the camera to a focus point and place the camera at a specific distance from that
    focus point. The camera stays in a direct line with the focus point.

    :param camera: the camera object
    :type camera: bpy.types.object
    :param focus_point: the point to focus on (default=``mathutils.Vector((0.0, 0.0, 0.0))``)
    :type focus_point: mathutils.Vector
    :param distance: the distance to keep to the focus point (default=``10.0``)
    :type distance: float
    """
    print("Moving camera")
    
    looking_direction = camera.location - focus_point
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()
    # Use * instead of @ for Blender <2.8
    camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))

# Functions to get annotaion bounding boxes 
def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = me_ob.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            # if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    mesh_eval.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    # Sanity check
    if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
        return (0, 0, 0, 0)

    return (
        round(min_x * dim_x),            # X
        round(dim_y - max_y * dim_y),    # Y
        round((max_x - min_x) * dim_x),  # Width
        round((max_y - min_y) * dim_y)   # Height
    )

# Render scene in JPEG format
def render_scene(it):
    bpy.context.scene.render.image_settings.file_format='JPEG'
    bpy.context.scene.render.filepath = output_dir_images + "/%0.5d.jpg"%it
    bpy.ops.render.render(use_viewport = True, write_still=True)

# Export annotations of bounding boxes in VOC format
def save_annotations(object, it):
    writer = Writer(output_dir_images + "/%0.5d.jpg"%it, r_settings.resolution_x, r_settings.resolution_y)
    if object is not None:
        bound_x, bound_y, bound_w, bound_h = (camera_view_bounds_2d(bpy.context.scene, bpy.context.scene.camera, object))
        part_name = str(object.name).split(".", 1)
        writer.addObject(part_name[0], bound_x, bound_y, bound_x+bound_w, bound_y+bound_h)
    writer.save(output_dir_annotations + "/%0.5d.xml"%it)

# PROGRAM CODE:

# Normalise color values. Blender requires colors to be define between 0-1. 
for c in colors0:
    c = [x/num for x in c]
    c.append(1) # add 4th element alpha = 1, in case PNG format is required
    colors.append(c)
    
# Create list of materials.
for i in range(len(colors)):
    color_name = "color_"+str(i)
    mat1 = bpy.data.materials.new(color_name)    
    mat1.diffuse_color = colors[i]
    r1 = random.randint(0,1)
    if r1 > 0:
        mat1.shadow_method = ("NONE")
    else:
        mat1.shadow_method = ("OPAQUE")
    materials.append(mat1)


# Create list of backgrounds
# Backgrounds are located in 'Collection 2' object container
for obj_bg in bpy.data.collections['Collection 2'].all_objects:
    if (obj_bg.type == 'MESH'):
        background.append(obj_bg)


# Create list of lego part objects 
for obj in bpy.data.collections['Collection'].all_objects:
    if (obj.type == 'MESH'):
        objects.append(obj)
        

# Create a table to count every object appearance in rendering batches
# Used to check if random selection of parts has a uniform distribution
# Prints the table after script is finished
for x in objects:
    object_count.append([x.name]) # Part name
 
for c in range(0,len(objects)):
    object_count[c].append(0) # Number of times part has been used


# Reset camera location and orientation towards an object
objects[0].select_set(True)
objects[0].location = (0,0,0.5)
objects[0].rotation_euler = (0, 0, 0)
bpy.data.objects['Camera'].location = (0,0,5)
bpy.data.objects['Camera'].rotation_euler = (0,0,0)
bpy.ops.view3d.camera_to_view_selected()
# If resolution not 300x300, adjust y (-1.0) to suit, otherwise object part might be out of the picture.
update_camera(bpy.data.objects['Camera'],focus_point=mathutils.Vector((0.0, -1.0, 0.5)), distance=4)


# Hide all objects and backgrounds
for x in objects:
    x.hide_set(True)
    x.hide_render = True

for bg in background:
    bg.hide_set(True)
    bg.hide_render = True

# GENERATING INDIVIDUAL PARTS RENDERS

# Render backgrounds
if (run_type == 1 or run_type == 4): 
    # Render few images without parts, only background 
    # This is not necessary (but recommended) since most object detection networks can handle empty scenes 
    for rr in range(0,3): 
        for bg in background:
            bg.hide_set(False) # Unhide one background
            bg.hide_render = False # Make it visible in renderings

            render_scene(start_range)
            save_annotations(None, start_range) # No object to annotate so passing None
            
            start_range += 1
            
            bg.hide_set(True)
            bg.hide_render = True     

# Render individual parts rotated around all axes
if (run_type == 1 or run_type == 3):
    # Set camera to default location for rendering individual parts
    bpy.data.objects['Camera'].location = (0,0,5)
    bpy.data.objects['Camera'].rotation_euler = (0,0,0)
    update_camera(bpy.data.objects['Camera'],focus_point=mathutils.Vector((0.0, -1.0, 0)), distance=2.5)

    for x in objects: 
        # Check if the part is not a dublicate so not to render twise same parts. Duplicates have suffix ".001"
        if '.' not in x.name: 
            x.hide_set(False)
            x.hide_render = False
            x.select_set(True)    
            
            # Randomise part location a bit
            x.location = (0,0,0.5)
            x.rotation_euler = (0, 0, 0)
            
            # Rotate around x
            for x_or in range(0,360,step_size):
                
                # Adjust part rotation
                x.rotation_euler = (math.radians(x_or), 0, 0)
                    
                # Choose material randomly
                r1 = random.randint(0, len(materials)-1) 
                x.active_material = materials[r1]
                
                # Hide all backgrounds    
                for bg in background:
                    bg.hide_set(True)
                    bg.hide_render = True
                    
                # Unhide one background randomly
                r3 = random.randint(0,len(background)-1)
                background[r3].hide_set(False)
                background[r3].hide_render = False     
                   
                render_scene(start_range)
                save_annotations(x, start_range)
            
                # Reset orientation & randommise location of the object
                x.location = (round(random.uniform(-0.15, 0.15), 4), round(random.uniform(-0.15, 0.15), 4), 0.5)
                x.rotation_euler = (0, 0, 0)

                # increase counter
                start_range += 1   
            
            x.location = (0,0,0.5)
            x.rotation_euler = (0, 0, 0)
            
            # Rotate around y
            for y_or in range(0,360,step_size):
                
                x.rotation_euler = (0, math.radians(y_or), 0)
         
                r1 = random.randint(0, len(materials)-1) 
                x.active_material = materials[r1]
                
                for bg in background:
                    bg.hide_set(True)
                    bg.hide_render = True
                    
                r3 = random.randint(0,len(background)-1)
                background[r3].hide_set(False)
                background[r3].hide_render = False     
                    
                render_scene(start_range)
                save_annotations(x, start_range)
                
                x.location = (round(random.uniform(-0.15, 0.15), 4), round(random.uniform(-0.15, 0.15), 4), 0.5)
                x.rotation_euler = (0, 0, 0)

                start_range += 1   
       
            x.location = (0,0,0.5)
            x.rotation_euler = (0, 0, 0)
            
            # Rotate around z
            for z_or in range(0,360,step_size):
                
                x.rotation_euler = (0, 0, math.radians(z_or))
         
                r1 = random.randint(0, len(materials)-1) 
                x.active_material = materials[r1]
                
                for bg in background:
                    bg.hide_set(True)
                    bg.hide_render = True
                    
                r3 = random.randint(0,len(background)-1)
                background[r3].hide_set(False)
                background[r3].hide_render = False     

                render_scene(start_range)
                save_annotations(x, start_range)
                
                x.location = (round(random.uniform(-0.15, 0.15), 4), round(random.uniform(-0.15, 0.15), 4), 0.5)
                x.rotation_euler = (0, 0, 0)

                start_range += 1   
            
            # Hide the part after finished rendering it's set
            x.hide_set(True)
            x.hide_render = True
            
    # END OF GENERATING INDIVIDUAL PARTS


# GENERATE BATCH OF LEGO & BACKGROUND RENDERINGS
if (run_type == 1 or run_type == 2):

    # Find the what is the number of existing rendered images in "images" directory.
    path, dirs, files = next(os.walk(output_dir+"/images"))
    file_count = len(files)
    start_range = file_count
    # start_range = 0 # Modify start_range to start from previous renders of individual parts
    image_set = start_range + batch_size

    print("Printing batch of images. Start_range from: ", start_range)
    
    # Deselect all objects
    for iii in range(start_range,image_set):
        
        bpy.ops.object.select_all(action='DESELECT')
        
        # Hide all objects
        for x in objects:
            x.hide_set(True)
            x.hide_render = True
            
        # Get a maximum number of 5 random lego part objects 
        list_of_objects_numbers = random.sample(range(len(objects)), random.randint(2, 5) )      
        print("Lego parts selected: ", list_of_objects_numbers)

        for obj_num in list_of_objects_numbers: 
            r1 = random.randint(0, len(materials)-1) 
            r2 = random.randint(0,1) # random number to decide in the part is in the scene
            
            x = objects[obj_num]

            # Unhide and select lego object that's in the list
            x.hide_set(False)
            x.hide_render = False
            x.select_set(True)

            # Move lego object to a random location within given constriants
            x.location = (round(random.uniform(-1.5, 1.5), 2), round(random.uniform(-1.5, 1.5), 2), 0.5)

            # Lego orientation randomisation
            x.rotation_euler = (math.radians(random.randint(0,180)), math.radians(random.randint(0,180)), math.radians(random.randint(0,180)))

            # Lego material randomisation
            x.active_material = materials[r1]
            
        # Update lego part counting table
        cc = 0
        for x in objects:
            if x.hide_render == False:
                object_count[cc][1] += 1
            cc += 1     
            
        # Background randomisation
        # Hide all backgrounds
        for bg in background:
            bg.hide_set(True)
            bg.hide_render = True
        
        # Unhide one background randomly
        r3 = random.randint(0,len(background)-1)
        background[r3].hide_set(False)
        background[r3].hide_render = False      
            
        # Fit camera scene within the objects
        bpy.ops.view3d.camera_to_view_selected()

        render_scene(iii)
        
        save_annotations(x, start_range)

        writer = Writer(output_dir_images + "/%0.5d.jpg"%iii, r_settings.resolution_x, r_settings.resolution_y)

        # Save annotations
        for x in objects:
            if x.hide_render == False:
                bound_x, bound_y, bound_w, bound_h = (camera_view_bounds_2d(bpy.context.scene, bpy.context.scene.camera, x))
                part_name = str(x.name).split(".", 1)
                # Save annotations of rectangle around the object: x_min, y_min, x_max, y_max
                writer.addObject(part_name[0], bound_x, bound_y, bound_x+bound_w, bound_y+bound_h)
                
        writer.save(output_dir_annotations + "/%0.5d.xml"%iii)

    # Print out times each Lego part was used
    print("Object count: ")
    for cc in object_count:
        print(cc)
    # END OF BATCH RENDERING CODE

print("All done.")

