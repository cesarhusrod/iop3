[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10995575.svg)](https://doi.org/10.5281/zenodo.10995575)

# IOP3

Pipeline para la reducción y extración de información de fuentes altamente polarizadas usando el instrumento CAFOS del Centro Astronómico Hispano-Alemán (CAHA).

## 1. Introducción

Este software permite procesar imágenes tomadas en el rango óptico de fuentes altamente polarizadas (BLAZARs). Son fuentes compactas, normalmente asociadas a un agujero negro, caracterizadas por la emisión de un chorro de partículas en dirección a la Tierra y con una alta variabilidad temporal.

Se emplea el instrumento CAFOS (*Calar Alto Faint Object Spectrograph*) (<https://www.caha.es/es/telescope-2-2m-2/cafos>), montado sobre el telescopio de 2.2 m de apertura del Observatorio del CALAR Alto (<https://www.caha.es/es/>).

Las imágenes generadas se observan a 4 ángulos de polarización diferentes(0, 22.5, 45 y 67.5º). En cada una de ellas, se generan dos imágenes de cada fuente (la imagen ordinaria y la extraordinaria). Realizando operaciones entre los flujos observados de ambas fuentes, y teniendo en cuenta los cuatro ángulos de polarización tomados en la misma noche, se puede obtener la polarización y el grado de polarización de cada fuente de estudio.

Haciendo un seguimiento de cada fuente de interés durante varias campañas que para algunas de ellas llevan años recopilando información, se puede hacer una idea precisa de la variabilidad temporal de cada una de ellas.

Todas estas medidas se almacenan durante el proceso en una base de datos MySQL, que es uno de los requisitos de instalación de la Pipeline IOP3.


## 2. Metodología

El procesamiento de las imágenes consiste en varios pasos:

0. **Búsqueda de targets cercanos al centro de la imagen** de ciencia y verificación de que el campo cubierto por esa imagen engloba a dicha fuente.

1. **Reducción de las imágenes**, mediante el procesado y escalado de la imagen de ciencia a partir de las imágenes de BIAS y FLATs en cada uno de los cuatro ángulos de polarización. Esta tarea implica la clasicación de imágenes contenidas en el directorio de entrada: BIAS, FLATs e imágenes de ciencia. También la generación de MasterBIAS y MasterFLATs.

2. **Rotado de imágenes** a 270º para orientarlas de forma adecuada.

3. **Localización y eliminación de cada una de las imágenes extraordinarias** para cada fuente de la imagen.

4. **Calibración astrométrica de la imagen**. Usando esta información, se sobreecribe la cabecera de la imagen reducida rotada. Esta parte implica la determinación en segundo grado del ángulo de la imagen resultante en los procesos anteriores con respecto a la amacenada en los catálogos. En algunos casos esa rotación llega hasta los cinco grados, lo que hacía originariamente imposible obtener una buena superposición entre fuentes detectadas en la imagen y registradas en los catálogos. 

5. **Detección de fuentes**. Localización de la fuente problema y de las fuentes que serán usadas como calibradores fotométricos en cada imagen. Para estos últimos, se seleccionan fuentes estables y no polarizadas.

7. **Calibración fotométrica** de la imagen. Se usa la magnitud de los calibradores y su flujo obtenido para determinar el punto cero fotométrico de la imagen en su conjunto.

8. Obtención de la **fotometría de las fuentes de interés**, tanto para la imagen ordinaria como la extraordinaria, en los cuatro filtros de estudio. El resultado se almacena en un fichero de salida.

9. **Procesado de los ficheros de salida** para almacenar todos los resultados intermedios (calibración astrométrica y fotométrica) y finales (polarización, ángulo de polarización y sus errores).
**Opcionalmente**, estos resultados pueden **agregarse a una base de datos MySQL** que facilitará la consulta de datos por fuente e intervalo temporal deseados al equipo de trabajo de *MAPCAT*. Para insertar información en la base de datos, deben ejecutarse algunos scripts PYTHON adicionales antes de ejecutar la pipeline. Corresponden a la creación de la base de datos (*iop3_create_database.py*) y rellenado de información previa referente a los blazars de estudio y sus calibradores fotométricos, así como las aperturas empleadas (*iop3_add_blazar_to_db.py*).

10. **Generación de gráficos y ficheros de datos** a partir de los ficheros de salida obtenidos por la Pipeline o la consulta a la base de datos. Esas figuras, gráficos e imágenes le será de mucha utilidad para la presentación de resultados en congresos y para ilustrar publicaciones científicas (artículos, posters...).

**El proceso distingue entre imágenes de fuentes principales de naturaleza estelar o galáctica**. Son imágenes con diferencias importantes a nivel de tiempo de observación y perfil de la matríz de la imagen.

Existen además **varios ficheros importantes que contienen información necesaria para la Pipeline**, como el filtrado de fuentes, su detección y la correcta asignación de aperturas y de magnitudes a la hora de hacer la calibración. Todos ellos están disponibles en la carpeta *conf*.

## 3. Descarga del código

Para poder hacerlo de forma sencilla, debe tener el [cliente GIT](https://git-scm.com/downloads) instalado en su ordenador.

Lo siguiente es abrir la consola de GIT (en el caso de trabajar en Windows) o la terminal (si lo hace en LINUX o MAC) y escribir la siguiente línea de código:

```git clone https://github.com/cesarhusrod/iop3.git```

Eso creará una carpeta en su ordenador en la directorio desde el que ha ejecutado el comando anterior de nombre *sarai_piezo_precip*. Incluye todo el código necesario y los ficheros de ejemplo.

La otra posibilidad consiste en descargar un paquete ZIP con todo el código a través de la siguiente dirección web: <https://github.com/cesarhusrod/iop3/archive/refs/heads/main.zip>


## 4. Instalación

El proceso de instalación es sencillo. Lo único que necesita es un intérprete de PYTHON e instalar las dependecias de la Pipeline, a saber:

* SExtractor (SOurce Extractor, <https://www.astromatic.net/software/sextractor/>). En algunas distribuciones como UBUNTU basta con ejecutar el siguiente comando para su instalación: apt-get install sextractor

* WCSTools (<http://tdc-www.harvard.edu/wcstools/>). El proceso es un poco largo, ya que tiene que definir algunas variables de entorno y descargarse los catálogos de USNO-A2 (en el rango óptico) y 2MASS (para el rango del infrarrojo cercano (bandas *J*, *H*, y *Ks*)). Ambos catálogos cubren todo el cielo, de manera que no habrá ningún apuntado que no pueda calibrar astrométricamente. La desventaja es que son un poco pesados en cuanto a tamaño.

* MySQL (<https://www.mysql.com/downloads/>). Es un sistema Gestor de Bases de Datos gratuito y multiplataforma. En las distribuciones LINUX su instalación es particularmente sencilla. En el caso de ubuntu sólo hay que ejecutar el comando: *apt-get install mysql*


## 5. Instrucciones de uso

La Pipeline IOP3 está diseñada para el trabajo de forma modular y para ficheros que se han obtenido durante una noche de observación. Entre dichos ficheros debería haber imágenes de FLATS y BIAS que se usarán para la reducción de las imágenes. Si las imágenes ya están reducidas o no es necesario realizar el proceso, se puede lanzar la Pipeline sin ellos.

Como se ha dicho, el trabajo de la Pipeline es **modular** y se estructura de forma **secuencial** tal y como se ha especificado en el apartado de metodología.

1. Reducción de las imágenes (módulo *iop3_reduction.py*)

2. Rotado de imágenes

3. Localización y eliminación de fuentes duplicadas

4. Calibración astrométrica de la imagen (módulo *iop3_astrometric_calibration.py*)

5. Detección de fuentes de calibración y obtención de sus flujos a una apertura dada

7. Calibración fotométrica de la imagen (módulo *iop3_photometric_calibration.py*)

8. Fotometría de las fuentes de estudio y cómputo de sus parámetros polarimétricos (módulo *iop3_polarimetry.py*)

9. Registro en la base de datos (opcional) (módulos *iop3_create_database.py*, *iop3_add_blazar_to_db.py*)


Se aplican en función de los parámetros pasados al script principal: *iop3_pipeline.py*

 * --border_image, valor entero. Corresponde al número de píxeles del borde de la imagen que no serán tenidos en cuenta en el proceso. Por defecto toma el valor de 15 píxeles.
 
 * --tol_pixs, valor entero. Es el valor de distancia máxima asumible entre la fuente detectada y la fuente del catálogo consultada para considerar que son la misma. Se usa en el proceso de calibración asrtométrica. El valor por defecto es de 15 píxeles. En nuestras peores calibraciones no se obtienen valores superiores a 3 píxeles. Dado que la escala es de 0.53" / pix, no no desviamos más de 1.5" en la identificación.

 * --ignore_farcalib. Si se escribe como parámetro del comando, no procesa la imagen si no hay una fuente de interés en el campo de visión.

 * --overwrite. Si se escribe como parámetro del comando sobreescribe calibraciones astrométricas y fotométricas previas en caso de haberse realizado. Si no, toma las previas como válidas y continúa el proceso. 

 * --skip_reduction. Si se escribe como parámetro del comando, no realiza el proceso de reducción de las imágenes.

 * --skip_astrocal. Si se escribe como parámetro del comando, no calibra astrométricamente las imágenes.

 * --skip_photometry. Si se escribe como parámetro del comando, no calibra fotométricamente las imágenes.

 * --skip_polarimetry. Si se escribe como parámetro del comando, no calcula la polarización ni el ángulo de polarización de las medidas tomadas de las fuentes ordinaria y extraordinaria de cada una de las cuatro exposiciones a cuatro ángulos de polarización.

 * --skip_db_registration. Si se escribe el parámetro como comando, no se realiza el registro de las medidas de polarización, ángulo de polarización y respectivos errores en la base de datos. Tampoco registra los parámetros de las cabeceras de imágenes ni sus estadísticas en la base de datos.


Como parámetros obligatorios:

* config_dir, que contiene la ruta al directorio que almacena los ficheros de configuración de la Pipeline.

* input_dir, que establece al ruta al directorio que contiene las imágenes obtenidas en la noche de observación (RAW).


Partiendo de que la política de MAPCAT al respecto de los directorios es crear un directorio para cada noche de observación (YYYYMMDD) y almacenar las imágenes obtenidas en un subdirectorio (directorio de entrada *Directorio_base/YYYYMMDD/raw*), los directorios de salida serán
*Directorio_base/YYYYMMDD/reduction* - para las imágenes reducidas
*Directorio_base/YYYYMMDD/calibration* - para las imágenes calibradas
*Directorio_base/YYYYMMDD/final* - para almacenar los resultados de polarimetría calculada a partir de los grupos de imágenes procesadas en los cuatro ángulos de polarización.

De esta forma se preserva la integridad de las imágenes originales (*raw*) y se puede modularizar el flujo de trabajo de la Pipeline. También, se dispone de todos los productos intermedios para un análisis detallado de los casos anómalos. La contrapartida es la mayor necesidad de espacio en disco para procesar cada noche de observación.

## 6. Agradecimientos

Este trabajo ha sido realizado por César Husillos Rodríguez, basado en la experiencia acumulada durante años de trabajo en el Instituto de Astrofisica de Andalucía en el desarrollo de otras Pipelines y trabajos realizados como Técnico Superior con cargo a proyectos.

En el desarrollo de este trabajo hay otras personas implicadas:

* Iván Agudo (*IAA-CSIC*) y Giacomo Bonnoli (*IAA-CSIC* & *Istituto Nazionale di Astrofisica*), que a lo largo del proceso de desarrollo han revisado resultados y asesorado al respecto de procedimientos alternativos para la verificación de resultados obtenidos en ciertas partes del proceso de la Pipeline.

* María Isabel Bernardos Martín (Doctora en Astrofísica que realizó uno de sus postdocs en el IAA-CSIC y Data Scientist, [Enlace Linkedin](https://es.linkedin.com/in/mar%C3%ADa-isabel-bernardos-mart%C3%ADn-5b71b1183)), que probó el código y ayudó a mejorarlo. Paralelamente, adaptó la pipeline IOP3 para el procesado de imágenes obtenidas en el Observatorio de Sierra Nevada (*OSN - IAA - CSIC*).

* Juan Escudero Pedrosa (Estudiante predoctoral *IAA-CSIC*), por la realización de gráficas que permitieron analizar los resultados obtenidos. También por sus sugerencias para la configuración de la infraestructura software de la Pipeline.


Nuestro agradecimiento al Observatorio del Calar Alto (*CAHA*), ya que se usaron sus instalaciones para obtener las imágenes como parte de las múltiples campañas de observaciones llevada a cabo por el programa *Monitoring AGN with Polarimetry at the Calar Alto Telescopes* ( *MAPCAT*, <https://home.iaa.csic.es/~iagudo/research/MAPCAT/MAPCAT.html>), liderado por científico titular Iván Agudo (IAA - CSIC).

La Pipeline hace uso de dos programas externos: SExtractor y WCSTools. Nuestro agraecimiento por el desarrollo de esos paquetes de software tan importantes y necesarios para el trabajo de la Pipeline IOP^3. Los citamos aquí en forma de publicaciones:

* "SExtractor: Software for source extraction.", E. Bertin and S. Arnouts, A&AS, 117:393–404, June 1996. [doi:10.1051/aas:1996164](https://doi.org/10.1051/aas:1996164).

* "WCSTools 4.0: Building Astrometry and Catalogs into Pipelines", Douglas J. Mink, in Astronomical Data Analysis Software and Systems XV, ASP Conference Series, Vol. 15, Edited by C. Gabriel, C. Arviset, D. Ponz, and E. Solano, San Francisco: Astronomical Society of the Pacific, 2006, p.204. ([WCTOOLS web page](http://tdc-www.harvard.edu/software/wcstools/index.html))


## 7. Contacto

César Husillos Rodríguez, actualmente Titulado Superior de OPIs en el Centro Nacional Instituto Geológico y Minero de España (CN IGME - CSIC, <https://www.igme.es>).

E-mail de trabajo: <c.husillos@igme.es>

También puedes contactar a través de *GitHub*.


## 8. Cómo citar este software

Si usa este software, por favor cítelo como sigue:

Cesar Husillos Rodriguez;María Isabel Bernardos Martín;Iván Agudo;Giacomo Bonnoli;Juan Escudero Pedrosa;2024, cesarhusrod/iop3: v1.0.0, Zenodo, DOI: 10.5281/zenodo.10995575, as developed on GitHub

## 9. Licencia

Este proyecto se ha desarrollado bajo la licencia GNU General Public License v3.0.


