from py_bing_search import PyBingWebSearch
import glob
import csv

def extract_snippet(prompt):
	API_KEY = "0nNf/RGQhw/62syJrJGDRbm4BUx4fwkyDYpiFLBobCo"
	bing_web = PyBingWebSearch(API_KEY, prompt, web_only=False)
	first_fifty_result= bing_web.search(limit=50, format='json')
	bing_result = []
	for result in first_fifty_result:
		bing_result.append(result.description)
	return bing_result

def extract_prompt(m_string):
	queries = dict()
	for big_text in m_string.split("\n"):
		queries[big_text.split(" ")[0]] = ' '.join(big_text.split(" ")[1:])
	return queries

if __name__ == '__main__':
	path = 'Essay_Set_Descriptions/'
	files = glob.glob(path+"/*.docx")

# bing_web = PyBingWebSearch('Your-Api-Key-Here', search_term, web_only=False)
# first_fifty_result= bing_web.search(limit=50, format='json') #1-50
# print (second_fifty_result[0].description)
	m_string= """1 effects computers have on people.
	2 censorship in libraries. books, music, movies, magazines remove if offensive
	3 features affect cyclist.
	4 resilience elasticity toughness
	5 mood author memoir
	6 obstacles Empire State Building dirigibles dock
	7 patience Being patient understanding tolerant
	8 benefits of laughter"""
	queries = extract_prompt(m_string)
	essay_set_bing_results = dict()
	for key in queries.keys():
		essay_set_bing_results[key] = extract_snippet(queries[key])

	w = csv.writer(open("dataset/bing_output.csv", "w"))
	for key, val in essay_set_bing_results.items():
		for v in val:
			print v
			w.writerow([key, v.encode('ascii', 'ignore').strip().decode('ascii')])
