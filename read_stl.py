import colorsys
import numpy as np

class stl_model(object):

    def __init__(self,file_path = './binary.stl'):
        # self.width = width
        # self.height = height
        # self.batch_size = batch_size
        self.path = file_path
        self.model = self.read_file(file_path)
        self.tri = self.creat_triangles()

    def read_file(self,path):
    
        normal = []
        vertex = []
        
        
        with open(path,'r') as f:
            
            while True:
                
                p = f.readline().strip()
                if p == 'endsolid':
                    break
                word = p.split()
                if word[0] == 'facet' and word[1] == 'normal':
                    x = float(word[2])
                    y = float(word[3])
                    z = float(word[4])
                    normal.append(None)
                    normal[len(normal)-1] = (x, y, z)
                elif word[0] == 'vertex':
                    x = float(word[1])/1000.
                    y = float(word[2])/1000.
                    z = float(word[3])/1000.
                    vertex.append(None)
                    vertex[len(vertex)-1] = (x, y, z)
                    
        assert len(normal) == len(vertex)/3
        
        return {"normal": normal,"vertex":vertex}  #normal是法向矢量

    def creat_triangles(self):
        
        normal = self.model['normal']
        vertex = self.model['vertex']
        
        tri_num = len(normal)
        
        nor_list = list(set(normal))
        nor_num = len(nor_list)
        
        # color = [ colorsys.hsv_to_rgb(np.random.uniform(),1.,1.) for i in range(nor_num)]
        # colors = [(a[0]*255,a[1]*255,a[2]*255) for a in color]
        
        # special_colors = [ colorsys.hsv_to_rgb(i/25.,1.,1.) for i in range(25)]
        # special_colors[0] =(0.1,0.1,0.1)
        # special_colors[24] = (1., 1., 1.)
        
        triangles = []
        
        for i in range(tri_num):
            nor = normal[i]
            p0 = vertex[i*3]
            p1 = vertex[i*3+1]
            p2 = vertex[i*3+2]
            # c = (255,255,255)
            red = (nor[0]+1)/2.
            green = (nor[1]+1)/2.
            blue = (nor[2]+1)/2.
            c = (red, green, blue)
            # for k in range(nor_num):
            #     Nor = nor_list[k]
            #     if Nor[0] == nor[0] and Nor[1] == nor[1] and Nor[2] == nor[2]:
            #         c = colors[k]
            #         break
            triangles.append(None)
            triangles[len(triangles)-1] = {"normal":nor,"p0":p0,"p1":p1,"p2":p2,"colors":c}
    
            
        return triangles
        
        
        





