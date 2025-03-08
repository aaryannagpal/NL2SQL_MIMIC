-- MySQL dump 10.13  Distrib 8.0.41, for Linux (x86_64)
--
-- Host: localhost    Database: capstone_mimic
-- ------------------------------------------------------
-- Server version	8.0.41

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `admissions`
--

DROP TABLE IF EXISTS `admissions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `admissions` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `admittime` datetime NOT NULL,
  `dischtime` datetime NOT NULL,
  `deathtime` datetime DEFAULT NULL,
  `admission_type` varchar(255) NOT NULL,
  `admission_location` varchar(255) DEFAULT NULL,
  `discharge_location` varchar(255) DEFAULT NULL,
  `insurance` varchar(255) NOT NULL,
  `language` varchar(255) NOT NULL,
  `marital_status` varchar(255) DEFAULT NULL,
  `ethnicity` varchar(255) NOT NULL,
  `edregtime` datetime DEFAULT NULL,
  `edouttime` datetime DEFAULT NULL,
  `hospital_expire_flag` tinyint(1) NOT NULL,
  UNIQUE KEY `admissions_idx04` (`hadm_id`),
  KEY `admissions_idx01` (`subject_id`,`hadm_id`),
  KEY `admissions_idx02` (`admittime`,`dischtime`,`deathtime`),
  KEY `admissions_idx03` (`admission_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `chartevents`
--

DROP TABLE IF EXISTS `chartevents`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `chartevents` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `stay_id` int unsigned NOT NULL,
  `charttime` datetime NOT NULL,
  `storetime` datetime DEFAULT NULL,
  `itemid` mediumint unsigned NOT NULL,
  `value` text,
  `valuenum` float DEFAULT NULL,
  `valueuom` varchar(255) DEFAULT NULL,
  `warning` tinyint(1) NOT NULL,
  KEY `chartevents_idx01` (`subject_id`,`hadm_id`,`stay_id`),
  KEY `chartevents_idx02` (`itemid`),
  KEY `chartevents_idx03` (`charttime`,`storetime`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3
/*!50100 PARTITION BY HASH (`itemid`)
PARTITIONS 50 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `d_hcpcs`
--

DROP TABLE IF EXISTS `d_hcpcs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `d_hcpcs` (
  `code` varchar(255) NOT NULL,
  `category` tinyint unsigned DEFAULT NULL,
  `long_description` text,
  `short_description` text NOT NULL,
  UNIQUE KEY `d_hcpcs_idx01` (`code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `d_icd_diagnoses`
--

DROP TABLE IF EXISTS `d_icd_diagnoses`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `d_icd_diagnoses` (
  `icd_code` varchar(255) NOT NULL,
  `icd_version` tinyint unsigned NOT NULL,
  `long_title` text NOT NULL,
  UNIQUE KEY `d_icd_diagnoses_icd_code_icd_version` (`icd_code`,`icd_version`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `d_icd_procedures`
--

DROP TABLE IF EXISTS `d_icd_procedures`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `d_icd_procedures` (
  `icd_code` varchar(255) NOT NULL,
  `icd_version` tinyint unsigned NOT NULL,
  `long_title` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `d_items`
--

DROP TABLE IF EXISTS `d_items`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `d_items` (
  `itemid` mediumint unsigned NOT NULL,
  `label` text NOT NULL,
  `abbreviation` varchar(255) NOT NULL,
  `linksto` varchar(255) NOT NULL,
  `category` varchar(255) NOT NULL,
  `unitname` varchar(255) DEFAULT NULL,
  `param_type` varchar(255) NOT NULL,
  `lownormalvalue` smallint DEFAULT NULL,
  `highnormalvalue` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `d_labitems`
--

DROP TABLE IF EXISTS `d_labitems`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `d_labitems` (
  `itemid` smallint unsigned NOT NULL,
  `label` varchar(255) DEFAULT NULL,
  `fluid` varchar(255) NOT NULL,
  `category` varchar(255) NOT NULL,
  `loinc_code` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `datetimeevents`
--

DROP TABLE IF EXISTS `datetimeevents`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `datetimeevents` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `stay_id` int unsigned NOT NULL,
  `charttime` datetime NOT NULL,
  `storetime` datetime NOT NULL,
  `itemid` mediumint unsigned NOT NULL,
  `value` datetime NOT NULL,
  `valueuom` varchar(255) NOT NULL,
  `warning` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `diagnoses_icd`
--

DROP TABLE IF EXISTS `diagnoses_icd`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `diagnoses_icd` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `seq_num` tinyint unsigned NOT NULL,
  `icd_code` varchar(255) NOT NULL,
  `icd_version` tinyint unsigned NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `drgcodes`
--

DROP TABLE IF EXISTS `drgcodes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `drgcodes` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `drg_type` varchar(255) NOT NULL,
  `drg_code` varchar(255) NOT NULL,
  `description` text,
  `drg_severity` tinyint unsigned DEFAULT NULL,
  `drg_mortality` tinyint unsigned DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `emar`
--

DROP TABLE IF EXISTS `emar`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `emar` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned DEFAULT NULL,
  `emar_id` varchar(255) NOT NULL,
  `emar_seq` smallint unsigned NOT NULL,
  `poe_id` varchar(255) NOT NULL,
  `pharmacy_id` int unsigned DEFAULT NULL,
  `charttime` datetime NOT NULL,
  `medication` varchar(255) DEFAULT NULL,
  `event_txt` varchar(255) DEFAULT NULL,
  `scheduletime` datetime DEFAULT NULL,
  `storetime` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `emar_detail`
--

DROP TABLE IF EXISTS `emar_detail`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `emar_detail` (
  `subject_id` int unsigned NOT NULL,
  `emar_id` varchar(255) NOT NULL,
  `emar_seq` smallint unsigned NOT NULL,
  `parent_field_ordinal` float DEFAULT NULL,
  `administration_type` varchar(255) DEFAULT NULL,
  `pharmacy_id` int unsigned DEFAULT NULL,
  `barcode_type` varchar(255) DEFAULT NULL,
  `reason_for_no_barcode` text,
  `complete_dose_not_given` varchar(255) DEFAULT NULL,
  `dose_due` varchar(255) DEFAULT NULL,
  `dose_due_unit` varchar(255) DEFAULT NULL,
  `dose_given` text,
  `dose_given_unit` varchar(255) DEFAULT NULL,
  `will_remainder_of_dose_be_given` varchar(255) DEFAULT NULL,
  `product_amount_given` varchar(255) DEFAULT NULL,
  `product_unit` varchar(255) DEFAULT NULL,
  `product_code` varchar(255) DEFAULT NULL,
  `product_description` text,
  `product_description_other` text,
  `prior_infusion_rate` varchar(255) DEFAULT NULL,
  `infusion_rate` varchar(255) DEFAULT NULL,
  `infusion_rate_adjustment` varchar(255) DEFAULT NULL,
  `infusion_rate_adjustment_amount` varchar(255) DEFAULT NULL,
  `infusion_rate_unit` varchar(255) DEFAULT NULL,
  `route` varchar(255) DEFAULT NULL,
  `infusion_complete` varchar(255) DEFAULT NULL,
  `completion_interval` varchar(255) DEFAULT NULL,
  `new_iv_bag_hung` varchar(255) DEFAULT NULL,
  `continued_infusion_in_other_location` varchar(255) DEFAULT NULL,
  `restart_interval` varchar(255) DEFAULT NULL,
  `side` varchar(255) DEFAULT NULL,
  `site` text,
  `non_formulary_visual_verification` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `hcpcsevents`
--

DROP TABLE IF EXISTS `hcpcsevents`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `hcpcsevents` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `chartdate` datetime NOT NULL,
  `hcpcs_cd` varchar(255) NOT NULL,
  `seq_num` tinyint unsigned NOT NULL,
  `short_description` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `icustays`
--

DROP TABLE IF EXISTS `icustays`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `icustays` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `stay_id` int unsigned NOT NULL,
  `first_careunit` varchar(255) NOT NULL,
  `last_careunit` varchar(255) NOT NULL,
  `intime` datetime NOT NULL,
  `outtime` datetime NOT NULL,
  `los` float NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `inputevents`
--

DROP TABLE IF EXISTS `inputevents`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `inputevents` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `stay_id` int unsigned NOT NULL,
  `starttime` datetime NOT NULL,
  `endtime` datetime NOT NULL,
  `storetime` datetime NOT NULL,
  `itemid` mediumint unsigned NOT NULL,
  `amount` float NOT NULL,
  `amountuom` varchar(255) NOT NULL,
  `rate` float DEFAULT NULL,
  `rateuom` varchar(255) DEFAULT NULL,
  `orderid` mediumint unsigned NOT NULL,
  `linkorderid` mediumint unsigned NOT NULL,
  `ordercategoryname` varchar(255) NOT NULL,
  `secondaryordercategoryname` varchar(255) DEFAULT NULL,
  `ordercomponenttypedescription` varchar(255) NOT NULL,
  `ordercategorydescription` varchar(255) NOT NULL,
  `patientweight` float NOT NULL,
  `totalamount` float DEFAULT NULL,
  `totalamountuom` varchar(255) DEFAULT NULL,
  `isopenbag` tinyint(1) NOT NULL,
  `continueinnextdept` tinyint(1) NOT NULL,
  `cancelreason` tinyint unsigned NOT NULL,
  `statusdescription` varchar(255) NOT NULL,
  `originalamount` float NOT NULL,
  `originalrate` float NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `labevents`
--

DROP TABLE IF EXISTS `labevents`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `labevents` (
  `labevent_id` int unsigned NOT NULL,
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned DEFAULT NULL,
  `specimen_id` int unsigned NOT NULL,
  `itemid` smallint unsigned NOT NULL,
  `charttime` datetime NOT NULL,
  `storetime` datetime DEFAULT NULL,
  `value` text,
  `valuenum` float DEFAULT NULL,
  `valueuom` varchar(255) DEFAULT NULL,
  `ref_range_lower` float DEFAULT NULL,
  `ref_range_upper` float DEFAULT NULL,
  `flag` varchar(255) DEFAULT NULL,
  `priority` varchar(255) DEFAULT NULL,
  `comments` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3
/*!50100 PARTITION BY HASH (`itemid`)
PARTITIONS 50 */;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `microbiologyevents`
--

DROP TABLE IF EXISTS `microbiologyevents`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `microbiologyevents` (
  `microevent_id` mediumint unsigned NOT NULL,
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned DEFAULT NULL,
  `micro_specimen_id` mediumint unsigned NOT NULL,
  `chartdate` datetime NOT NULL,
  `charttime` datetime DEFAULT NULL,
  `spec_itemid` mediumint unsigned NOT NULL,
  `spec_type_desc` varchar(255) NOT NULL,
  `test_seq` tinyint unsigned NOT NULL,
  `storedate` datetime DEFAULT NULL,
  `storetime` datetime DEFAULT NULL,
  `test_itemid` mediumint unsigned NOT NULL,
  `test_name` varchar(255) NOT NULL,
  `org_itemid` mediumint unsigned DEFAULT NULL,
  `org_name` varchar(255) DEFAULT NULL,
  `isolate_num` tinyint unsigned DEFAULT NULL,
  `quantity` varchar(255) DEFAULT NULL,
  `ab_itemid` mediumint unsigned DEFAULT NULL,
  `ab_name` varchar(255) DEFAULT NULL,
  `dilution_text` varchar(255) DEFAULT NULL,
  `dilution_comparison` varchar(255) DEFAULT NULL,
  `dilution_value` float DEFAULT NULL,
  `interpretation` varchar(255) DEFAULT NULL,
  `comments` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `outputevents`
--

DROP TABLE IF EXISTS `outputevents`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `outputevents` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `stay_id` int unsigned NOT NULL,
  `charttime` datetime NOT NULL,
  `storetime` datetime NOT NULL,
  `itemid` mediumint unsigned NOT NULL,
  `value` float NOT NULL,
  `valueuom` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `patients`
--

DROP TABLE IF EXISTS `patients`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `patients` (
  `subject_id` int unsigned NOT NULL,
  `gender` varchar(255) NOT NULL,
  `anchor_age` tinyint unsigned NOT NULL,
  `anchor_year` smallint unsigned NOT NULL,
  `anchor_year_group` varchar(255) NOT NULL,
  `dod` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pharmacy`
--

DROP TABLE IF EXISTS `pharmacy`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `pharmacy` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `pharmacy_id` int unsigned NOT NULL,
  `poe_id` varchar(255) DEFAULT NULL,
  `starttime` datetime DEFAULT NULL,
  `stoptime` datetime DEFAULT NULL,
  `medication` varchar(255) DEFAULT NULL,
  `proc_type` varchar(255) NOT NULL,
  `status` varchar(255) NOT NULL,
  `entertime` datetime NOT NULL,
  `verifiedtime` datetime DEFAULT NULL,
  `route` varchar(255) DEFAULT NULL,
  `frequency` varchar(255) DEFAULT NULL,
  `disp_sched` varchar(255) DEFAULT NULL,
  `infusion_type` varchar(255) DEFAULT NULL,
  `sliding_scale` varchar(255) DEFAULT NULL,
  `lockout_interval` varchar(255) DEFAULT NULL,
  `basal_rate` float DEFAULT NULL,
  `one_hr_max` varchar(255) DEFAULT NULL,
  `doses_per_24_hrs` tinyint unsigned DEFAULT NULL,
  `duration` float DEFAULT NULL,
  `duration_interval` varchar(255) DEFAULT NULL,
  `expiration_value` smallint unsigned DEFAULT NULL,
  `expiration_unit` varchar(255) DEFAULT NULL,
  `expirationdate` datetime DEFAULT NULL,
  `dispensation` varchar(255) DEFAULT NULL,
  `fill_quantity` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `poe`
--

DROP TABLE IF EXISTS `poe`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `poe` (
  `poe_id` varchar(255) NOT NULL,
  `poe_seq` smallint unsigned NOT NULL,
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `ordertime` datetime NOT NULL,
  `order_type` varchar(255) NOT NULL,
  `order_subtype` varchar(255) DEFAULT NULL,
  `transaction_type` varchar(255) NOT NULL,
  `discontinue_of_poe_id` varchar(255) DEFAULT NULL,
  `discontinued_by_poe_id` varchar(255) DEFAULT NULL,
  `order_status` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `poe_detail`
--

DROP TABLE IF EXISTS `poe_detail`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `poe_detail` (
  `poe_id` varchar(255) NOT NULL,
  `poe_seq` smallint unsigned NOT NULL,
  `subject_id` int unsigned NOT NULL,
  `field_name` varchar(255) NOT NULL,
  `field_value` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `prescriptions`
--

DROP TABLE IF EXISTS `prescriptions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `prescriptions` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `pharmacy_id` int unsigned NOT NULL,
  `starttime` datetime DEFAULT NULL,
  `stoptime` datetime DEFAULT NULL,
  `drug_type` varchar(255) NOT NULL,
  `drug` varchar(255) DEFAULT NULL,
  `gsn` text,
  `ndc` varchar(255) DEFAULT NULL,
  `prod_strength` text,
  `form_rx` varchar(255) DEFAULT NULL,
  `dose_val_rx` varchar(255) DEFAULT NULL,
  `dose_unit_rx` varchar(255) DEFAULT NULL,
  `form_val_disp` varchar(255) DEFAULT NULL,
  `form_unit_disp` varchar(255) DEFAULT NULL,
  `doses_per_24_hrs` tinyint unsigned DEFAULT NULL,
  `route` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `procedureevents`
--

DROP TABLE IF EXISTS `procedureevents`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `procedureevents` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `stay_id` int unsigned NOT NULL,
  `starttime` datetime NOT NULL,
  `endtime` datetime NOT NULL,
  `storetime` datetime NOT NULL,
  `itemid` mediumint unsigned NOT NULL,
  `value` float NOT NULL,
  `valueuom` varchar(255) NOT NULL,
  `location` varchar(255) DEFAULT NULL,
  `locationcategory` varchar(255) DEFAULT NULL,
  `orderid` mediumint unsigned NOT NULL,
  `linkorderid` mediumint unsigned NOT NULL,
  `ordercategoryname` varchar(255) NOT NULL,
  `secondaryordercategoryname` varchar(255) DEFAULT NULL,
  `ordercategorydescription` varchar(255) NOT NULL,
  `patientweight` float NOT NULL,
  `totalamount` varchar(255) DEFAULT NULL,
  `totalamountuom` varchar(255) DEFAULT NULL,
  `isopenbag` tinyint(1) NOT NULL,
  `continueinnextdept` tinyint(1) NOT NULL,
  `cancelreason` tinyint(1) NOT NULL,
  `statusdescription` varchar(255) NOT NULL,
  `comments_date` varchar(255) DEFAULT NULL,
  `originalamount` float NOT NULL,
  `originalrate` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `procedures_icd`
--

DROP TABLE IF EXISTS `procedures_icd`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `procedures_icd` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `seq_num` tinyint unsigned NOT NULL,
  `chartdate` datetime NOT NULL,
  `icd_code` varchar(255) NOT NULL,
  `icd_version` tinyint unsigned NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `services`
--

DROP TABLE IF EXISTS `services`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `services` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned NOT NULL,
  `transfertime` datetime NOT NULL,
  `prev_service` varchar(255) DEFAULT NULL,
  `curr_service` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `transfers`
--

DROP TABLE IF EXISTS `transfers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `transfers` (
  `subject_id` int unsigned NOT NULL,
  `hadm_id` int unsigned DEFAULT NULL,
  `transfer_id` int unsigned NOT NULL,
  `eventtype` varchar(255) NOT NULL,
  `careunit` varchar(255) DEFAULT NULL,
  `intime` datetime NOT NULL,
  `outtime` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-02-20 12:26:18
