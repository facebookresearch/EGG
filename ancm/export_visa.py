import os
import pandas
from collections import defaultdict
import xml.etree.ElementTree as ET

directory = 'visa_dataset/UK'
concepts = {}
homonyms = []
for file in os.listdir(directory):
    if not os.path.isfile(os.path.join(directory, file)):
        continue
    tree = ET.parse(os.path.join(directory, file))

    for concept in tree.getroot().iter('concept'):
        concept_dict = defaultdict(int)
        for category in concept:
            attributes = [a.strip() for a in category.text.split()
                          if a.strip()]  # and not a.startswith('beh')]
            for a in attributes:
                concept_dict[a] = 1
        if concept.attrib['name'].find('_(') == -1:
            concept_name = concept.attrib['name']
        else:
            concept_name = concept.attrib['name'][:concept.attrib['name'].find('_(')]

        if concept_name not in concepts and concept_name not in homonyms:
            concepts[concept_name] = concept_dict
        elif concept_name in homonyms:
            print('skipping', concept_name)
        else:
            del concepts[concept_name]
            homonyms.append(concept_name)
            print('deleting', concept_name)

all_attributes = set([k for cd in concepts.values() for k in cd.keys()])
output_dict = {'concept': list(concepts.keys())}
for attribute in all_attributes:
    output_dict[attribute] = [cd[attribute] for cd in concepts.values()]
output = pandas.DataFrame(output_dict)
output.to_csv('visa.csv', index=False)

print('number of concepts:', len(output))
print('number of attributes:', len(output.columns) - 1)



#print(concepts)
#    for subcategory in tree.getroot()[:1]:
#        for concept in subcategory:
#            concept_dict = defaultdict(int)
#            for attr_category in concept:
#                if not attr_category or not attr_category.text.strip():
#                    print(concept, attr_category)
#                print([attr.strip() for attr in attr_category.text.split('\n') if attr.strip()])
                #for attr in attr_category.text.split('\n'):
                #    if attr.tag.startswith('beh'):
                #        continue

