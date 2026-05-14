"""
Implementación simulador robotico: Esta es una versión adaptada del simulador propuesto originalmente por: Paul O'Dowd y Hemma Philamore
https://github.com/paulodowd/GoogleColab_Simple2DSimulator
"""


from math import *
import numpy as np
from random import shuffle, randrange
import sys
import matplotlib.pyplot as plt
np.random.seed(42)

"""
Creación de la clase que define los obstaculos. En nuestros escenarios los obstaculos tendran forma circular.
"""

class Obstacle_wall:
    # Se asigna una posición aleatoria dentro del entorno manteniendo una distancia con el centro (donde inicia el robot)
    def __init__(self, x_prop, y_prop, arena_size=200, radius=1, rot=0.0, max_obstacles=1):
        self.radius = radius

        rot_ang = rot * ((np.pi * 2) / max_obstacles)
        rand_dist = np.random.uniform(.35, .85) * (arena_size / 2)
        # self.x = (arena_size/2) + rand_dist*np.cos(rot_ang)
        # self.y = (arena_size/2) + rand_dist*np.sin(rot_ang)
        self.x = x_prop
        self.y = y_prop

class Obstacle_c:
  # Se asigna una posición aleatoria dentro del entorno manteniendo una distancia con el centro (donde inicia el robot)
  def __init__(self, x_prop, y_prop, arena_size=200, radius=1, rot=0.0, max_obstacles=1):

    self.radius = radius


    rot_ang = rot * ((np.pi*2)/max_obstacles)

    rand_dist = np.random.uniform(0.1, .7) * (arena_size/2)
    #print('arena size',arena_size/2)
    if x_prop<0 and y_prop<0:
      self.x = (arena_size/2) + rand_dist*np.cos(rot_ang)
      self.y = (arena_size/2) + rand_dist*np.sin(rot_ang)
    else:
      self.x = x_prop
      self.y = y_prop





"""
Clase que define el funcionamiento de cada sensor de proximidad.
"""
class ProxSensor_c:

  # Posición global en xy del sensor
  x = 0
  y = 0
  theta = 0

  # Para guardar la ultima lectura
  reading = 0

  # Localizar el sensor alrededor del cuerpo del robot
  offset_dist = 0
  offset_angl = 0

  # maximo rango de lectura en cm
  max_range = 20


  def __init__(self, offset_dist=5, offset_angl=0):
    self.offset_dist = offset_dist
    self.offset_angl = offset_angl


  def updateGlobalPosition(self, robot_x, robot_y, robot_theta ):

    # Dirección actual del sensor es la rotación del robot
    # mas la rotación predeterminada del sensor en relación
    # al cuerpo del robot.
    self.theta = self.offset_angl + robot_theta

    sensor_x = (self.offset_dist*np.cos(self.theta))
    sensor_y = (self.offset_dist*np.sin(self.theta))


    self.x = sensor_x + robot_x
    self.y = sensor_y + robot_y

    self.reading = -1

  def scanFor( self, obstruction ):

    # Escanear por posibles obstaculos
    distance = np.sqrt( ((obstruction.x - self.x)**2) + ((obstruction.y - self.y)**2) )
    distance = distance - obstruction.radius


    # Si esta fuera de rango, no retorna nada
    if distance > self.max_range:
      return

    a2o = atan2( ( obstruction.y - self.y), (obstruction.x-self.x ))

    angle_between = atan2( sin(self.theta-a2o),  cos(self.theta-a2o) )
    angle_between = abs( angle_between )


    if angle_between > np.pi/8:
      return


    if self.reading < 0:
      self.reading = distance


    if self.reading > 0:
      if distance < self.reading:
        self.reading = distance





"""
Clase que define el funcionamiento del robot
"""
class Robot_c:


  def __init__(self, x=50,y=50,theta=np.pi, objetivo = [190,190,5]):
    self.x = x
    self.y = y
    self.theta = np.pi/4# theta
    self.stall = -1 # evaluar colisiones

    self.desire = -1
    self.score = 0
    self.radius = 5 # 5cm radio
    self.wheel_sep = self.radius*2 # Mismo tamaño de rueda a cada lado
    self.vl = 0
    self.vr = 0
    self.objetivo = objetivo
    # ubicación de los sensores en radianes
    self.sensor_dirs = [5.986479,
                        5.410521,
                        4.712389,
                        3.665191,
                        2.617994,
                        1.570796,
                        0.8726646,
                        0.296706,
                        ]

    self.prox_sensors = []
    for i in range(0,8):
      self.prox_sensors.append( ProxSensor_c(self.radius, self.sensor_dirs[i]) )


  def updatePosition( self, vl, vr ):

    if vl > 1.0:
      vl = 1.0
    if vl < -1.0:
      vl = -1.0
    if vr > 1.0:
      vr = 1.0
    if vr < -1.0:
      vr = -1.0

    self.vl = vl
    self.vr = vr

    self.stall = -1

    # Matriz del movimiento del robot
    r_matrix = [(vl/2)+(vr/2),0, (vr-vl)/self.wheel_sep]

    # Matriz para convertir referencia local a referencia global
    k_matrix = [
                [ np.cos(self.theta),-np.sin(self.theta),0],
                [ np.sin(self.theta), np.cos(self.theta),0],
                [0,0,1]
               ]

    result_matrix = np.matmul( k_matrix, r_matrix)

    self.x += result_matrix[0]
    self.y += result_matrix[1]
    self.theta += result_matrix[2]

    for prox_sensor in self.prox_sensors:
      prox_sensor.updateGlobalPosition( self.x, self.y, self.theta )




  def updateSensors(self, obstruction ):

    for prox_sensor in self.prox_sensors:
      prox_sensor.scanFor( obstruction )

  def collisionCheck(self, obstruction ):
    distance = np.sqrt( ((obstruction.x - self.x)**2) + ((obstruction.y - self.y)**2) )
    distance -= self.radius
    distance -= obstruction.radius
    if distance < 0:
      self.stall = 1
      angle = atan2( obstruction.y - self.y, obstruction.x - self.x)
      self.x += distance * np.cos(angle)
      self.y += distance * np.sin(angle)

  def updateScore(self):
    # Se define la metrica a usar para evaluar el desempeño.
    new_score = 0
    if self.desire == -1:
        if abs(self.x -self.objetivo[0]) < self.objetivo[2] and abs(self.y -self.objetivo[1]) < self.objetivo[2] :
            new_score = 100
            self.desire = 1
        if self.stall == 1:
            new_score -= 5
    self.score += new_score

#Función para la generación de paredes
def crear_paredes_v2(trayectoria, num_puntos_por_segmento=50):
    puntos_totales = []

    if len(trayectoria) < 2:
        return trayectoria

    for i in range(len(trayectoria) - 1):
        p_inicio = np.array(trayectoria[i])
        p_fin = np.array(trayectoria[i + 1])

        for t in np.linspace(0, 1, num_puntos_por_segmento, endpoint=False):
            punto_intermedio = p_inicio + t * (p_fin - p_inicio)
            puntos_totales.append(punto_intermedio.tolist())

    puntos_totales.append(trayectoria[-1])

    return puntos_totales
