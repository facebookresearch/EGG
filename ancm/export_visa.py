import os
import pandas
from collections import defaultdict
import xml.etree.ElementTree as ET

directory = 'data/visa_dataset/UK'
concepts = {}
homonyms = []
for file in os.listdir(directory):
    if not os.path.isfile(os.path.join(directory, file)):
        continue
    tree = ET.parse(os.path.join(directory, file))

    concept_category = tree.getroot().attrib['category']
    for concept in tree.getroot().iter('concept'):
        attribute_dict = defaultdict(int)
        for category in concept:
            attributes = [a.strip() for a in category.text.split()
                          if a.strip()]  # and not a.startswith('beh')]
            for a in attributes:
                attribute_dict[a] = 1
        if concept.attrib['name'].find('_(') == -1:
            concept_name = concept.attrib['name']
        else:
            concept_name = concept.attrib['name'][:concept.attrib['name'].find('_(')]

        if concept_name not in concepts and concept_name not in homonyms:
            concepts[concept_name] = {
                'category': concept_category,
                'attributes': attribute_dict,
            }
        elif concept_name in homonyms:
            print('skipping', concept_name)
        else:
             del concepts[concept_name]
             homonyms.append(concept_name)
             print('deleting', concept_name)

all_attributes = set([k for cd in concepts.values() for k in cd['attributes'].keys()])
output_dict = {
    'concept': list(concepts.keys()),
    'category': [cd['category'] for cd in concepts.values()],
}
for attribute in all_attributes:
    output_dict[attribute] = [cd['attributes'][attribute] for cd in concepts.values()]
output = pandas.DataFrame(output_dict)
os.makedirs('data', exist_ok=True)
output.to_csv('data/visa.csv', index=False)

print('number of concepts:', len(output))
print('number of attributes:', len(output.columns) - 2)
