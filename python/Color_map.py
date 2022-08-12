class Color_map:
    def __init__(self,data_matrix):

        self.__color_data_matrix = data_matrix

    
    def get_color_data_matrix(self):
        return self.__color_data_matrix
    
    def set_color(self, y_ind, x_ind, new_color):
        self.__color_data_matrix[y_ind, x_ind] = new_color
