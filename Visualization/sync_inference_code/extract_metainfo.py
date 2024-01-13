import pydicom
import json

# Re-reading the DICOM SEG file after environment reset
dicom_seg_path = '/mnt/sda/freelance_project_girgis/Visualization/example_data/Segmentation 1'

def extract_segmentation_info(dicom_seg_path):
    dcm = pydicom.dcmread(dicom_seg_path)

    segment_attributes = []
    if 'SegmentSequence' in dcm:
        for segment in dcm.SegmentSequence:
            segment_info = {
                "labelID": segment.SegmentNumber,
                "SegmentDescription": getattr(segment, 'SegmentDescription', ''),
                "SegmentLabel": getattr(segment, 'SegmentLabel', ''),
                "SegmentAlgorithmType": getattr(segment, 'SegmentAlgorithmType', ''),
                "SegmentAlgorithmName": getattr(segment, 'SegmentAlgorithmName', ''),
            }

            # Extract properties from SegmentedPropertyCategoryCodeSequence and SegmentedPropertyTypeCodeSequence
            category = segment.SegmentedPropertyCategoryCodeSequence[0]
            segment_info["SegmentedPropertyCategoryCodeSequence"] = {
                "CodeValue": category.CodeValue,
                "CodingSchemeDesignator": category.CodingSchemeDesignator,
                "CodeMeaning": category.CodeMeaning
            }

            type_code = segment.SegmentedPropertyTypeCodeSequence[0]
            segment_info["SegmentedPropertyTypeCodeSequence"] = {
                "CodeValue": type_code.CodeValue,
                "CodingSchemeDesignator": type_code.CodingSchemeDesignator,
                "CodeMeaning": type_code.CodeMeaning
            }

            # Extract recommendedDisplayRGBValue if available
            if hasattr(segment, 'RecommendedDisplayCIELabValue'):
                lab_values = segment.RecommendedDisplayCIELabValue
                # Convert CIELab to RGB (placeholder, actual conversion may be more complex)
                rgb_values = [int(v / 65535 * 255) for v in lab_values[:3]]  # Simplified conversion
                segment_info["recommendedDisplayRGBValue"] = rgb_values

            segment_attributes.append([segment_info])

    return {"segmentAttributes": segment_attributes}

# Extract segmentation info and save as JSON
segmentation_info = extract_segmentation_info(dicom_seg_path)
metainfo_json_path = '/mnt/sda/freelance_project_girgis/Visualization/sync_inference_code/metainfo_2.json'
with open(metainfo_json_path, 'w') as f:
    json.dump(segmentation_info, f, indent=4)