from sentence_transformers import SentenceTransformer, util
import numpy as np
import statistics

model = SentenceTransformer('stsb-roberta-large')



message = [{'content 1': 'I believe that taxes should be cut. Taxpayers should instead receive excess state funds.'}, {'content 2': 'I want schools to be open even if there are COVID-19 outbreaks. Children will suffer under online learning.'}, {'content 3': 'We need more funding for our school systems. I want teachers to be paid more and to encourage professions in education.'}, {'content 4': 'I want to protect fetuses. I am strongly against abortion'}, {'content 5': 'I want elections to be secure. No cheating should be allowed'}]
message2 = [{'content 1': 'I want all citizens to have affordable housing, and to find places to house the homeless.'}, {'content 2': 'I want to protect the environment against climate change and pollution. '}, {'content 3': 'I want guns to be regulated. I am pro gun-control'}, {'content 4': 'I want healthcare to be more affordable, and for Medicaid to be expanded'}, {'content 5': 'I want more people to vote. Voting should be expanded, not restricted'}]
sampleusers = [message, message2]
with open('staceyabrams.txt', 'r') as file:
    s1 = file.read().replace('\n', '')
    s1 = s1.replace('\xa0', '')


with open('briankemp.txt', 'r') as file:
    s2 = file.read().replace('\n', '')

similarities = []
similarphrases = []
for user in sampleusers:
    similaritystacey = []
    similaritykemp = []
    mostsimilarphrasestacey = []
    mostsimilarphrasekemp = []
    for content in user:
        #print(content)
        #print(type(content))
        for key in content:
            viewpoint = content.get(key)
            viewpoint_embedding = model.encode(viewpoint, convert_to_tensor=True)
            #embedding2 = model.encode(s1, convert_to_tensor=True)
            #embedding3 = model.encode(s2, convert_to_tensor=True)

            corpus = s1.split(".")
            corpus_embeddings = model.encode(corpus, convert_to_tensor=True)


            cos_scores = util.pytorch_cos_sim(viewpoint_embedding, corpus_embeddings)[0]
        # Sort the results in decreasing order and get the first top_k
            top_results = np.argpartition(-cos_scores, range(1))[0:1]
            #print("Sentence:", viewpoint, "\n")
            #print("Top", 1, "most similar sentences in corpus:")

            staceybest = ''
            for idx in top_results[0:1]:
                #print(corpus[idx], "(Score: %.4f)" % (cos_scores[idx]))
                similaritystacey.append(cos_scores[idx])
                staceybest = corpus[idx]
            

            corpus2 = s2.split(".")
            corpus_embeddings2 = model.encode(corpus2, convert_to_tensor=True)
            cos_scores2 = util.pytorch_cos_sim(viewpoint_embedding, corpus_embeddings2)[0]
            top_results2 = np.argpartition(-cos_scores2, range(1))[0:1]
            kempbest = ''
            for idx in top_results2[0:1]:
                            #print(corpus[idx], "(Score: %.4f)" % (cos_scores[idx]))
                similaritykemp.append(cos_scores2[idx])
                kempbest = corpus2[idx]

            if user == message2:
                mostsimilarphrasestacey.append(staceybest)
            else:
                mostsimilarphrasekemp.append(kempbest)

            

            




    similarities.append([similaritystacey, similaritykemp])
    similarphrases.append([mostsimilarphrasestacey, mostsimilarphrasekemp])
            

            
            #cosine_scorestacey = util.pytorch_cos_sim(embedding1, embedding2)
            #cosine_scorekemp = util.pytorch_cos_sim(embedding1, embedding3)

            #similaritystacey.append(cosine_scorestacey)
            #similaritykemp.append(cosine_scorekemp)
    #print(similaritystacey)
    #print(similaritykemp)

print(similarphrases)

similarities = [[[0.4497, 0.3175, 0.4840, 0.3919, 0.4812], [0.4872, 0.4761, 0.6298, 0.5347, 0.6132]], [[0.6341, 0.6189, 0.5840, 0.6191, 0.6288], [0.3819, 0.2714, 0.3909, 0.5655, 0.3197]]]



for i in range(2):
    for j in range(2):
        similarities[i][j] = statistics.mean(similarities[i][j])
        


for i in range(2):
    if similarities[i][0] > similarities[i][1]:
        similarities[i] = "Stacey Abrams"
    else:
        similarities[i] = "Brian Kemp"

kemp_voter = ['I believe that taxes should be cut. Taxpayers should instead receive excess state funds.', 'I want schools to be open even if there are COVID-19 outbreaks. Children will suffer under online learning.', 'We need more funding for our school systems. I want teachers to be paid more and to encourage professions in education.', 'I want to protect fetuses. I am strongly against abortion', 'I want elections to be secure. No cheating should be allowed']
stacey_voter = ['I want all citizens to have affordable housing, and to find places to house the homeless.', 'I want to protect the environment against climate change and pollution.', 'I want guns to be regulated. I am pro gun-control', 'I want healthcare to be more affordable, and for Medicaid to be expanded', 'I want more people to vote. Voting should be expanded, not restricted']

kemp_best = ['Mirroring the rebate passed earlier this year, this proposal – if passed by the General Assembly – will use nearly $1 billion in funds from the state budget surplus to return money to Georgians’ pockets– because that is your money, not the government’s', 'While the overwhelming majority of school systems across the state did the right thing by getting their students back in the classroom as soon as possible, a select few did not', ' The need for more qualified, passionate teachers in our classrooms and counselors in our schools has never been greater', 'Signed the historic Heartbeat Bill (HB 481) to protect the unborn', 'Banned outside money from flooding local elections, expanded early voting days, secured drop boxes around the clock, and required photo ID for absentee ballots among other provisions to secure our elections']
stacey_best = ['I will ensure Georgians can access stable, safe, affordable housing by tackling housing affordability, the shrinking inventory of housing, the displacement of longtime residents due to gentrification, and the preventable tragedy of homelessness', 'My environmental action plan will protect our air, water and vulnerable communities, prepare us for extreme weather events, and generate significant job growth through advanced energy, innovative technologies and energy efficiency efforts', ' I will support universal background checks and limiting access to guns for perpetrators of domestic violence and stalking', 'I plan to expand Medicaid to lower health care costs for all, provide half a million Georgians with insurance and stop hospital closures', 'Protecting and expanding access to our fundamental freedom to vote is my imperative']

kemp = [['1. Brian Kemp', kemp_voter, kemp_best], ['2. Stacey Abrams']]
stacey = [['1. Stacey Abrams', stacey_voter, stacey_best], ['2. Brian Kemp']]
print('\n\n')
print(kemp)
print('\n\n')
print(stacey)