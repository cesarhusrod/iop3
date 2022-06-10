-- CREATE DATABASE IF NOT EXISTS `iop3db`;

-- USE `iop3db`;

DROP TABLE IF EXISTS `blazar_polarimetry`;
DROP TABLE IF EXISTS `polarimetry`;
DROP TABLE IF EXISTS `photometry`;
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
  `aper_pix_sext` int unsigned DEFAULT NULL COMMENT 'Fotometry aperture for SExtractor (pixels)',
  `name` varchar(50) NOT NULL COMMENT 'Source name',
  `name_IAU` varchar(50) DEFAULT NULL COMMENT 'IAU source name',
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
  -- `fwhm` float DEFAULT NULL COMMENT 'Source FWHM (pixels)',
  -- `fwhm_std` float DEFAULT NULL COMMENT 'Source FWHM error (pixels)',
  -- `fwhm_nsources` float DEFAULT NULL COMMENT 'Number of sources used in FWHM computation',
  -- `fwhm_flag` float DEFAULT NULL COMMENT 'Greatest SExtractor FLAG for sources used in FWHM computation',
  -- `fwhm_ellip` float DEFAULT NULL COMMENT 'Greatest SExtractor ELLIPTICITY for sources used in FWHM computation',
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
  `pol_angle` DECIMAL(6, 2) DEFAULT NULL COMMENT 'Grism angle',
  `crotation` DECIMAL(5, 2) DEFAULT NULL COMMENT 'Astrocal N-S FITS rotation angle [degrees]',
  `fwhm` DECIMAL(5, 2) DEFAULT NULL COMMENT 'Source FWHM (pixels)',
  `fwhm_std` DECIMAL(5, 2) DEFAULT NULL COMMENT 'Source FWHM error (pixels)',
  `fwhm_nsources` int DEFAULT NULL COMMENT 'Number of sources used in FWHM computation',
  `fwhm_flag` int DEFAULT NULL COMMENT 'Greatest SExtractor FLAG for sources used in FWHM computation',
  `fwhm_ellip` DECIMAL(5, 2) DEFAULT NULL COMMENT 'Greatest SExtractor ELLIPTICITY for sources used in FWHM computation',
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
  `wcstol` DECIMAL(5, 2) DEFAULT NULL COMMENT 'Matching tolerance (pixels)',
  `crota1` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Rotation angle in dimension 1 (degrees)',
  `crota2` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Rotation angle in dimension 2 (degrees)',
  `secpix1` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Pixel scale in dimension 1 (arcsecs/pix)',
  `secpix2` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Pixel scale in dimension 1 (arcsecs/pix)',
  `wcssep` float DEFAULT NULL COMMENT 'Mean distance between catalog and image detected sources (pixels)',
  `imwcs` varchar(250) DEFAULT NULL COMMENT 'Software and version used for astrometric calibration',
  `mag_zpt` DECIMAL(15, 8) DEFAULT NULL COMMENT 'Magnitude zero-point',
  `mag_zpt_std` DECIMAL(10,5) NOT NULL  COMMENT 'Standar deviation for magnitude zero-point',
  `ns_zpt` int DEFAULT NULL COMMENT 'Number of sources employed in mag_zpt estimation',
  `max` DECIMAL(8, 2) DEFAULT NULL COMMENT 'Maximum pixel value',
  `min` DECIMAL(8, 2) DEFAULT NULL COMMENT 'Minimum pixel value',
  `mean` DECIMAL(8, 2) DEFAULT NULL COMMENT 'Mean pixel value',
  `std` DECIMAL(8, 2) DEFAULT NULL COMMENT 'Standar deviation pixel value',
  `median` DECIMAL(8, 2) DEFAULT NULL COMMENT 'Median pixel value',
  FOREIGN KEY (`blazar_id`) REFERENCES `blazar_source`(`id`) on delete cascade on update cascade,
  FOREIGN KEY (`reduced_id`) REFERENCES `image_reduced`(`id`) on delete cascade on update cascade
);


CREATE TABLE IF NOT EXISTS `photometry`(
  -- `id` int unsigned NOT NULL AUTO_INCREMENT,
  `cal_id` int unsigned NOT NULL,
  `blazar_id` int unsigned NOT NULL,
  `date_run` date DEFAULT NULL COMMENT 'Night date when raw image was taken',
  `date_obs` datetime DEFAULT NULL COMMENT 'UT date at beginning integration',
  `mjd_obs` DECIMAL(16, 8) NOT NULL COMMENT 'Modified Julian Date for third image for each group of observations ordered by ascending date',
  `rjd-50000` DECIMAL(16, 8) NOT NULL COMMENT 'Reduced Julian Date for third image for each group of observations ordered by ascending date',
  -- `type` varchar(1) NOT NULL,
  `pol_angle` DECIMAL(6, 2) DEFAULT NULL COMMENT 'Grism angle',
  `aperpix` DECIMAL(5, 2) DEFAULT NULL COMMENT 'Photometry aperture in pixels',
  `fwhm` DECIMAL(5, 2) DEFAULT NULL COMMENT 'Estimated FITS FWHM (pixels)',
  `secpix` DECIMAL(5, 2) DEFAULT NULL COMMENT 'Computed arcsec / pix in FITS after astrocalibration',
  `exptime` DECIMAL(6, 2) DEFAULT NULL COMMENT 'Exposure FITS time (seconds)',
  `magzpt` DECIMAL(6, 2) DEFAULT NULL COMMENT 'Magnitude zeropoint for FITS after photocalibration',
  `mag_auto` DECIMAL(10, 4) NOT NULL COMMENT 'Source magnitude given by optimal aperture (AB)',
  `magerr_auto` DECIMAL(10, 4) NOT NULL COMMENT 'Source error mag_auto magnitude (AB)',
  `flux_auto` DECIMAL(15, 8) NOT NULL COMMENT 'Source flux given optimal aperture (counts)',
  `fluxerr_auto` DECIMAL(15, 8) NOT NULL COMMENT 'Source flux error given flux_aper aperture (counts)',
  `flux_aper` DECIMAL(15, 8) NOT NULL COMMENT 'Source flux given circular fixed aperture (counts)',
  `fluxerr_aper` DECIMAL(15, 8) NOT NULL COMMENT 'Source error flux given circular fixed aperture (counts)',
  `mag_aper` DECIMAL(10, 4) NOT NULL COMMENT 'Source magnitude given circular fixed aperture (AB)',
  `magerr_aper` DECIMAL(10, 4) NOT NULL COMMENT 'Source error magnitude given circular fixed aperture (AB)',
  `x_image` DECIMAL(10, 2) NOT NULL COMMENT 'X source coordinate (pixels)',
  `y_image` DECIMAL(10, 2) NOT NULL COMMENT 'Y source coordinate (pixels)',
  `alpha_j2000` DECIMAL(15, 8) NOT NULL COMMENT 'Source Right Ascension (degrees)',
  `delta_j2000` DECIMAL(15, 8) NOT NULL COMMENT 'Source Declination (degrees)',
  `flags` int DEFAULT NULL COMMENT 'SExtractor FLAG assigned to source',
  `class_star` DECIMAL(7, 6) NOT NULL COMMENT 'SExtractor classification (1: star, 0: non-star)',
  `fwhm_image` DECIMAL(5, 2) DEFAULT NULL COMMENT 'Estimated source FWHM (pixels)', 
  `fwhm_world` DECIMAL(5, 2) DEFAULT NULL COMMENT 'Estimated source FWHM (arcsecs)', 
  `elongation` DECIMAL(10, 4) NOT NULL COMMENT 'SExtractor elongation measure (1: circular source)',
  `ellipticity` DECIMAL(10, 4) NOT NULL COMMENT 'SExtractor ellipticity measure (0: circular source)',
  `distance_deg` DECIMAL(12, 10) NOT NULL COMMENT 'Distance between closest blazar and source (degrees)',
  `source_type` VARCHAR(20) NOT NULL COMMENT 'Source type: O (ordinary), E (extraordinary)',
  PRIMARY KEY (`cal_id`, `source_type`),
  FOREIGN KEY (`blazar_id`) REFERENCES `blazar_source`(`id`) on delete cascade on update cascade,
  FOREIGN KEY (`cal_id`) REFERENCES `image_calibrated`(`id`) on delete cascade on update cascade
);

CREATE TABLE IF NOT EXISTS `polarimetry`(
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
  `Q` DECIMAL(10, 4) NOT NULL COMMENT 'Stokes parameter Q',
  `dQ` DECIMAL(10, 4) NOT NULL COMMENT 'Error in Stokes parameter Q',
  `U` DECIMAL(10, 4) NOT NULL COMMENT 'Stokes parameter U',
  `dU` DECIMAL(10, 4) NOT NULL COMMENT 'Error in Stokes parameter U',
  `exptime` DECIMAL(6, 2) NOT NULL COMMENT 'Exposure FITS time (seconds)',
  `aperpix` DECIMAL(5, 2) NOT NULL COMMENT 'Photometry aperture in pixels',
  `aperas` DECIMAL(8, 6) NOT NULL COMMENT 'Photometry aperture in arcsecs',
  `num_angles` int NOT NULL COMMENT 'Number of measurements used in polarimetry computation',
  FOREIGN KEY (`blazar_id`) REFERENCES `blazar_source`(`id`) on delete cascade on update cascade,
  -- Only one measure for date_run and blazar
  UNIQUE (`rjd-50000`)
);


-- This table set relation between blazar_photometry for each image_calibrated and final P+-dP/Theta+-dTheta stored in polarimetry
-- CREATE TABLE IF NOT EXISTS `blazar_polarimetry`(
--   `cal_id` int unsigned NOT NULL,
--   `pol_id` int unsigned NOT NULL,
--   FOREIGN KEY (`cal_id`) REFERENCES `image_calibrated`(`id`) on delete cascade on update cascade,
--   FOREIGN KEY (`pol_id`) REFERENCES `polarimetry`(`id`) on delete cascade on update cascade,
--   PRIMARY KEY (`cal_id`, `pol_id`)
-- );