-- CREATE DATABASE IF NOT EXISTS `iop3db`;

-- USE `iop3db`;

DROP TABLE IF EXISTS `blazar_polarimetry`;
DROP TABLE IF EXISTS `polarimetry_data`;
DROP TABLE IF EXISTS `blazar_measure`;
DROP TABLE IF EXISTS `image_calibrated`;
DROP TABLE IF EXISTS `image_reduced`;
DROP TABLE IF EXISTS `raw_flat`;
DROP TABLE IF EXISTS `master_flat`;
DROP TABLE IF EXISTS `raw_bias`;
DROP TABLE IF EXISTS `master_bias`;
DROP TABLE IF EXISTS `image_raw`;
DROP TABLE IF EXISTS `blazar_source`;


CREATE TABLE IF NOT EXISTS `blazar_source`(
  `id` int unsigned NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT 'Object identification',
  `source_id` int unsigned DEFAULT NULL COMMENT 'Associated Blazar identification for this object',
  `aper_pix_sext` int unsigned NOT NULL COMMENT 'Fotometry aperture for SExtractor (pixels)',
  `name` varchar(50) NOT NULL COMMENT 'Source name',
  `name_IAU` varchar(50) NOT NULL COMMENT 'IAU source name',
  `ra2000` varchar(50) NOT NULL COMMENT 'Right Ascension (hours minutes seconds.)',
  `dec2000` varchar(50) NOT NULL  COMMENT 'Declination (degrees minutes seconds)',
  `rmag` DECIMAL(10,5) DEFAULT NULL COMMENT 'Magnitude in R filter for source (AB)',
  `rmagerr` DECIMAL(10,5) DEFAULT NULL COMMENT 'Error magnitude in R filter for source (AB)',
  `P` DECIMAL(10,5) DEFAULT NULL COMMENT 'Polarization measured for this source',
  `dP` DECIMAL(10,5) DEFAULT NULL COMMENT 'Error in Polarization measured for this source',
  `Theta` DECIMAL(10,5) DEFAULT NULL COMMENT 'Polarization Angle measured for this source',
  `dTheta` DECIMAL(10,5) DEFAULT NULL COMMENT 'Error in Polarization Angle measured for this source'
);


CREATE TABLE IF NOT EXISTS `image_raw`(
  `id` int unsigned NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT 'Raw image identificator',
  `date_run` date NOT NULL COMMENT 'Night date were image were taken',
  `path` varchar(100) NOT NULL COMMENT 'File name',
  `naxis1` int DEFAULT NULL COMMENT 'Number of pixels in dimension 1',
  `naxis2` int DEFAULT NULL COMMENT 'Number of pixels in dimension 2',
  `object` varchar(50) DEFAULT NULL COMMENT 'Object name',
  `type` varchar(30) DEFAULT NULL COMMENT 'bias, flat, science...',
  `ra2000` DECIMAL(15, 8) NOT NULL COMMENT 'Right Ascension (degrees)',
  `dec2000` DECIMAL(15,8) NOT NULL COMMENT 'Declination (degrees)',
  `exptime` float NOT NULL  COMMENT 'Exposure time (seconds)',
  `date_obs` datetime NOT NULL COMMENT 'UT date at beginning integration',
  `equinox` float DEFAULT NULL COMMENT 'Equinox (years)',
  `mjd_obs` varchar(30) NOT NULL COMMENT 'Modified Julian Date of observation',
  `pixscale` float DEFAULT NULL COMMENT 'Pixel scale in arcs',
  `filtname` varchar(50) DEFAULT NULL COMMENT 'Filter name',
  `telescope` varchar(50) NOT NULL COMMENT 'Telescope name',
  `instrument` varchar(50) NOT NULL COMMENT 'Instrument name',
  `pol_angle` float DEFAULT NULL COMMENT 'Grism angle',
  `max` float DEFAULT NULL COMMENT 'Maximum pixel value',
  `min` float DEFAULT NULL COMMENT 'Minimum pixel value',
  `mean` float DEFAULT NULL COMMENT 'Mean pixel value',
  `std` float DEFAULT NULL COMMENT 'Standar deviation pixel value',
  `median` float DEFAULT NULL COMMENT 'Median pixel value'
);


CREATE TABLE IF NOT EXISTS `master_bias`(
  `id` int unsigned NOT NULL PRIMARY KEY AUTO_INCREMENT  COMMENT 'MasterBIAS identificator',
  `date_run` date NOT NULL COMMENT 'Night date were bias used for generating masterBIAS were taken',
  `path` varchar(100) DEFAULT NULL COMMENT 'File name',
  `naxis1` int DEFAULT NULL COMMENT 'Number of pixels in dimension 1',
  `naxis2` int DEFAULT NULL COMMENT 'Number of pixels in dimension 2',
  `type` varchar(20) DEFAULT NULL COMMENT 'Always masterBIAS',
  `soft` varchar(150) DEFAULT NULL COMMENT 'Pipeline name and version',
  `proc_date` datetime DEFAULT NULL COMMENT 'Date when masterBIAS was generated',
  `pix_border` int DEFAULT NULL COMMENT 'Number of pixels discarded in image borders',
  `bias_operation` varchar(25) DEFAULT NULL COMMENT 'Operation used for combining raw bias data',
  `max` float DEFAULT NULL COMMENT 'Maximum pixel value',
  `min` float DEFAULT NULL COMMENT 'Minimum pixel value',
  `mean` float DEFAULT NULL COMMENT 'Mean pixel value',
  `std` float DEFAULT NULL COMMENT 'Standar deviation pixel value',
  `median` float DEFAULT NULL COMMENT 'Median pixel value'
);


CREATE TABLE IF NOT EXISTS `raw_bias`(
  `id`int unsigned NOT NULL PRIMARY KEY AUTO_INCREMENT,
  `master_bias_id` int unsigned NOT NULL,
  `raw_id` int unsigned NOT NULL,
  FOREIGN KEY (`master_bias_id`) REFERENCES `master_bias`(`id`) on delete cascade on update cascade,
  FOREIGN KEY (`raw_id`) REFERENCES `image_raw`(`id`) on delete cascade on update cascade
);



CREATE TABLE IF NOT EXISTS `master_flat`(
  `id` int unsigned NOT NULL PRIMARY KEY AUTO_INCREMENT,
  `master_bias_id` int unsigned NOT NULL COMMENT 'MasteBIAS id used in masterFLAT generation',
  `date_run` date NOT NULL COMMENT 'Night date were flats used for generating masterFLAT were taken',
  `path` varchar(100) NOT NULL COMMENT 'File name',
  `naxis1` int DEFAULT NULL COMMENT 'Number of pixels in dimension 1',
  `naxis2` int DEFAULT NULL COMMENT 'Number of pixels in dimension 2',
  `type` varchar(30) DEFAULT NULL COMMENT 'Always masterBIAS',
  -- `date_obs` date DEFAULT NULL,
  `pol_angle` float DEFAULT NULL COMMENT 'Grism angle',
  `soft` varchar(150) DEFAULT NULL COMMENT 'Pipeline name and version',
  `proc_date` datetime DEFAULT NULL COMMENT 'Date and time when masterFLAT was generated',
  `pix_border` int DEFAULT NULL COMMENT 'Number of pixels discarded in image borders',
  `flat_operation` varchar(25) DEFAULT NULL COMMENT 'Operation used for combining raw flats data',
  -- `object` varchar(40) DEFAULT NULL,
  `max` float DEFAULT NULL COMMENT 'Maximum pixel value',
  `min` float DEFAULT NULL COMMENT 'Minimum pixel value',
  `mean` float DEFAULT NULL COMMENT 'Mean pixel value',
  `std` float DEFAULT NULL COMMENT 'Standar deviation pixel value',
  `median` float DEFAULT NULL COMMENT 'Median pixel value',
  FOREIGN KEY (`master_bias_id`) REFERENCES `master_bias`(`id`) on delete cascade on update cascade
);


CREATE TABLE IF NOT EXISTS `raw_flat`(
  `id`int unsigned NOT NULL PRIMARY KEY AUTO_INCREMENT,
  `master_flat_id` int unsigned NOT NULL,
  `raw_id` int unsigned NOT NULL,
  FOREIGN KEY (`master_flat_id`) REFERENCES `master_flat`(`id`) on delete cascade on update cascade,
  FOREIGN KEY (`raw_id`) REFERENCES `image_raw`(`id`) on delete cascade on update cascade
);

CREATE TABLE IF NOT EXISTS `image_reduced`(
  `id` int unsigned NOT NULL PRIMARY KEY AUTO_INCREMENT,
  `raw_id` int unsigned NOT NULL,
  `master_bias_id` int unsigned NOT NULL,
  `master_flat_id` int unsigned NOT NULL,
  `path` varchar(100) NOT NULL COMMENT 'File name',
  `soft` varchar(100) DEFAULT NULL COMMENT 'Pipeline name and version',
  `date_run` date DEFAULT NULL COMMENT 'Night date when raw image was taken',
  `proc_date` date DEFAULT NULL COMMENT 'Date and time when reduced image was generated',
  `pix_border` int DEFAULT NULL COMMENT 'Number of pixels discarded in image borders',
  `naxis1` int DEFAULT NULL COMMENT 'Number of pixels in dimension 1',
  `naxis2` int DEFAULT NULL COMMENT 'Number of pixels in dimension 2',
  `type` varchar(30) DEFAULT NULL COMMENT 'bias, flat, science...',
  `object` varchar(30) NOT NULL COMMENT 'Object name',
  `ra2000` DECIMAL(15, 8) NOT NULL COMMENT 'Right Ascension (degrees)',
  `dec2000` DECIMAL(15, 8) NOT NULL COMMENT 'Declination (degrees)',
  `exptime` float NOT NULL  COMMENT 'Exposure time (seconds)',
  `date_obs` datetime NOT NULL COMMENT 'UT date at beginning integration',
  `equinox` float DEFAULT NULL COMMENT 'Equinox (years)',
  `mjd_obs` varchar(30) NOT NULL COMMENT 'Modified Julian Date of observation',
  `pixscale` float DEFAULT NULL COMMENT 'Pixel scale in arcs',
  `filtname` varchar(50) DEFAULT NULL COMMENT 'Filter name',
  `telescope` varchar(50) NOT NULL COMMENT 'Telescope name',
  `instrument` varchar(50) NOT NULL COMMENT 'Instrument name',
  `pol_angle` float DEFAULT NULL COMMENT 'Grism angle',
  `fwhm` float DEFAULT NULL COMMENT 'Source FWHM (pixels)',
  `fwhm_std` float DEFAULT NULL COMMENT 'Source FWHM error (pixels)',
  `fwhm_nsources` float DEFAULT NULL COMMENT 'Number of sources used in FWHM computation',
  `fwhm_flag` float DEFAULT NULL COMMENT 'Greatest SExtractor FLAG for sources used in FWHM computation',
  `fwhm_ellip` float DEFAULT NULL COMMENT 'Greatest SExtractor ELLIPTICITY for sources used in FWHM computation',
  `max` float DEFAULT NULL COMMENT 'Maximum pixel value',
  `min` float DEFAULT NULL COMMENT 'Minimum pixel value',
  `mean` float DEFAULT NULL COMMENT 'Mean pixel value',
  `std` float DEFAULT NULL COMMENT 'Standar deviation pixel value',
  `median` float DEFAULT NULL COMMENT 'Median pixel value',
  FOREIGN KEY (`master_bias_id`) REFERENCES `master_bias`(`id`) on delete cascade on update cascade,
  FOREIGN KEY (`master_flat_id`) REFERENCES `master_flat`(`id`) on delete cascade on update cascade,
  FOREIGN KEY (`raw_id`) REFERENCES `image_raw`(`id`) on delete cascade on update cascade
);


CREATE TABLE IF NOT EXISTS `image_calibrated`(
  `id` int unsigned NOT NULL PRIMARY KEY AUTO_INCREMENT,
  `reduced_id` int unsigned NOT NULL COMMENT 'Reduced image identification',
  -- `master_bias_id` int NOT NULL,
  -- `master_flat_id` int NOT NULL,
  `blazar_id` int unsigned NOT NULL COMMENT 'Blazar identification',
  `path` varchar(100) DEFAULT NULL COMMENT 'File name',
  `soft` varchar(100) DEFAULT NULL COMMENT 'Pipeline name and version',
  `date_run` date DEFAULT NULL COMMENT 'Night date when raw image was taken',
  `proc_date` date DEFAULT NULL  COMMENT 'Date and time when calibrated image was generated',
  `naxis1` int DEFAULT NULL COMMENT 'Number of pixels in dimension 1',
  `naxis2` int DEFAULT NULL COMMENT 'Number of pixels in dimension 2',
  `pix_border` int DEFAULT NULL COMMENT 'Number of pixels discarded in image borders',
  `object` varchar(50) DEFAULT NULL COMMENT 'Object name',
  `type` varchar(30) DEFAULT NULL COMMENT 'bias, flat, science...',
  `ra2000` varchar(30) NOT NULL COMMENT 'Right Ascension (hours:minutes:seconds)',
  `dec2000` varchar(30) NOT NULL COMMENT 'Declination (degrees:minutes:seconds)',
  `exptime` float NOT NULL  COMMENT 'Exposure time (seconds)',
  `date_obs` datetime NOT NULL COMMENT 'UT date at beginning integration',
  `equinox` float DEFAULT NULL COMMENT 'Equinox (years)',
  `mjd_obs` varchar(30) NOT NULL COMMENT 'Modified Julian Date of observation',
  `pixscale` float DEFAULT NULL COMMENT 'Pixel scale in arcs',
  `filtname` varchar(50) DEFAULT NULL COMMENT 'Filter name',
  `telescope` varchar(50) NOT NULL COMMENT 'Telescope name',
  `instrument` varchar(50) NOT NULL COMMENT 'Instrument name',
  `pol_angle` float DEFAULT NULL COMMENT 'Grism angle',
  `fwhm` float DEFAULT NULL COMMENT 'Source FWHM (pixels)',
  `fwhm_std` float DEFAULT NULL COMMENT 'Source FWHM error (pixels)',
  `fwhm_nsources` float DEFAULT NULL COMMENT 'Number of sources used in FWHM computation',
  `fwhm_flag` float DEFAULT NULL COMMENT 'Greatest SExtractor FLAG for sources used in FWHM computation',
  `fwhm_ellip` float DEFAULT NULL COMMENT 'Greatest SExtractor ELLIPTICITY for sources used in FWHM computation',
  `softdet` varchar(100) DEFAULT NULL COMMENT 'Software employed in source detection',
  -- Calibration parameters
  `crval1` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Astrometrical Right Aascension of image (degrees)',
  `crval2` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Astrometrical Declination of image (degrees)',
  `epoch` float DEFAULT NULL COMMENT 'Epoch',
  `crpix1` DECIMAL(15, 8) DEFAULT NULL,
  `crpix2` DECIMAL(15, 8) DEFAULT NULL,
  `cdelt1` DECIMAL(15, 8) DEFAULT NULL,
  `cdelt2` DECIMAL(15, 8) DEFAULT NULL,
  `ctype1` varchar(25) DEFAULT NULL COMMENT 'Astrometric solution type',
  `ctype2` varchar(25) DEFAULT NULL COMMENT 'Astrometric solution type',
  `cd1_1` DECIMAL(15, 8) DEFAULT NULL,
  `cd1_2` DECIMAL(15, 8) DEFAULT NULL,
  `cd2_1` DECIMAL(15, 8) DEFAULT NULL,
  `cd2_2` DECIMAL(15, 8) DEFAULT NULL,
  `wcsrefcat` varchar(20) DEFAULT NULL COMMENT 'Reference catalog for astrometry calibration',
  `wcsmatch` int DEFAULT NULL COMMENT 'Number of matched sources',
  `wcsnref` int DEFAULT NULL COMMENT 'Total number of sources proposed for matching',
  `wcstol` float DEFAULT NULL COMMENT 'Matching tolerance (pixels)',
  `crota1` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Rotation angle in dimension 1 (degrees)',
  `crota2` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Rotation angle in dimension 2 (degrees)',
  `secpix1` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Pixel scale in dimension 1 (arcsecs/pix)',
  `secpix2` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Pixel scale in dimension 1 (arcsecs/pix)',
  `wcssep` float DEFAULT NULL COMMENT 'Mean distance between catalog and image detected sources (pixels)',
  `imwcs` varchar(250) DEFAULT NULL COMMENT 'Software and version used for astrometric calibration',
  `mag_zpt` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Magnitude zero-point',
  `mag_zpt_std` DECIMAL(10,5) NOT NULL  COMMENT 'Standar deviation for magnitude zero-point',
  `ns_zpt` int DEFAULT NULL COMMENT 'Number of sources employed in mag_zpt estimation',
  `max` float DEFAULT NULL COMMENT 'Maximum pixel value',
  `min` float DEFAULT NULL COMMENT 'Minimum pixel value',
  `mean` float DEFAULT NULL COMMENT 'Mean pixel value',
  `std` float DEFAULT NULL COMMENT 'Standar deviation pixel value',
  `median` float DEFAULT NULL COMMENT 'Median pixel value',
  FOREIGN KEY (`blazar_id`) REFERENCES `blazar_source`(`id`) on delete cascade on update cascade,
  FOREIGN KEY (`reduced_id`) REFERENCES `image_reduced`(`id`) on delete cascade on update cascade
);


CREATE TABLE IF NOT EXISTS `blazar_measure`(
  -- `id` int unsigned NOT NULL AUTO_INCREMENT,
  `cal_id` int unsigned NOT NULL,
  `blazar_id` int unsigned NOT NULL,
  `date_run` date DEFAULT NULL COMMENT 'Night date when raw image was taken',
  `date_obs` datetime DEFAULT NULL COMMENT 'UT date at beginning integration',
  `mjd_obs` DECIMAL(16, 8) NOT NULL COMMENT 'Modified Julian Date for third image for each group of observations ordered by ascending date',
  -- `type` varchar(1) NOT NULL,
  `pol_angle` float DEFAULT NULL COMMENT 'Grism angle',
  `source_type` VARCHAR(20) NOT NULL COMMENT 'Source type: O (ordinary), E (extraordinary)',
  `object` varchar(50) NOT NULL COMMENT 'Object name',
  `fwhm` float DEFAULT NULL COMMENT 'Source FWHM (pixels)',
  `ra2000` DECIMAL(15, 8) NOT NULL COMMENT 'Source Right Ascension (degrees)',
  `dec2000` DECIMAL(15, 8) NOT NULL COMMENT 'Source Declination (degrees)',
  `flux_max` DECIMAL(15, 8) NOT NULL COMMENT 'Maximun flux in any source pixel (counts)',
  `flux_aper` DECIMAL(15, 8) NOT NULL COMMENT 'Source flux given circular fixed aperture (counts)',
  `fluxerr_aper` DECIMAL(15, 8) NOT NULL COMMENT 'Source error flux given circular fixed aperture (counts)',
  `mag_aper` DECIMAL(10, 4) NOT NULL COMMENT 'Source magnitude given circular fixed aperture (AB)',
  `magerr_aper` DECIMAL(10, 4) NOT NULL COMMENT 'Source error magnitude given circular fixed aperture (AB)',
  `class_star` DECIMAL(7, 6) NOT NULL COMMENT 'SExtractor classification (1: star, 0: non-star)',
  `ellipticity` DECIMAL(10, 4) NOT NULL COMMENT 'SExtractor ellipticity measure (0: circular source)',
  `flags` int DEFAULT NULL COMMENT 'SExtractor FLAG assigned to source',
  PRIMARY KEY (`cal_id`, `source_type`),
  FOREIGN KEY (`blazar_id`) REFERENCES `blazar_source`(`id`) on delete cascade on update cascade,
  FOREIGN KEY (`cal_id`) REFERENCES `image_calibrated`(`id`) on delete cascade on update cascade
);


CREATE TABLE IF NOT EXISTS `polarimetry_data`(
  `id` int unsigned NOT NULL PRIMARY KEY AUTO_INCREMENT,
  `blazar_id` int unsigned NOT NULL COMMENT 'Blazar id',
  `date_run` date NOT NULL COMMENT 'Night date were images used in computation were taken',
  `rjd-50000` DECIMAL(16, 8) NOT NULL COMMENT 'Modified Julian Date minus 50000 for third image for each group of observations',
  `name` varchar(50) DEFAULT NULL COMMENT 'Blazar name',
  `P` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Polarization (%)',
  `dP` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Polarization error (%)',
  `Theta` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Polarization Angle (degrees)',
  `dTheta` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Error in polarization Angle (degrees)',
  `R` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Computed magnitude (AB)',
  `dR` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Computed error in magnitude (AB)',
  FOREIGN KEY (`blazar_id`) REFERENCES `blazar_source`(`id`) on delete cascade on update cascade,
  -- Only one measure for date_run and blazar
  UNIQUE (`rjd-50000`)
);


-- This table set relation between blazar_measure for each image_calibrated and final P+-dP/Theta+-dTheta stored in polarimetry_data
-- CREATE TABLE IF NOT EXISTS `blazar_polarimetry`(
--   `cal_id` int unsigned NOT NULL,
--   `pol_id` int unsigned NOT NULL,
--   FOREIGN KEY (`cal_id`) REFERENCES `image_calibrated`(`id`) on delete cascade on update cascade,
--   FOREIGN KEY (`pol_id`) REFERENCES `polarimetry_data`(`id`) on delete cascade on update cascade,
--   PRIMARY KEY (`cal_id`, `pol_id`)
-- );