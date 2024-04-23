# %% [markdown]
# Following this URL: https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/5_Levels_Of_Text_Splitting.ipynb
# 
# Probally need to install the following package:
# pip install --upgrade  openai tiktoken langchain weaviate-client plotly nbformat
# 
# OPENAI_API_KEY key will need to be set e.g. creating a cell with:
# %env OPENAI_API_KEY=sk-FILLMEIN

# %%
# Grab a collection of data based on either keywords or semantic search
import pandas as pd
import weaviate #! Ai powered search engine. Fetches and processes data as well as handling ML tasks
import weaviate.classes as wvc
import weaviate.classes.query as wvcq
from weaviate.exceptions import WeaviateQueryError


import time
from dateutil import parser
from datetime import datetime, timezone


def grab_collection(client,concept,isKeyword,limit,batch,offset):
  # print(f"start {concept}")
#   seconds = time.time()
  # client = weaviate.connect_to_local(host="10.68.169.67", port=8080)
  
  col = client.collections.get("Universal_project")
  
#   collection = col.query.hybrid(
#       query=concept,
#       alpha=0.05,
#       fusion_type=wvcq.HybridFusion.RELATIVE_SCORE,
#       auto_limit=100,
#       limit=1000,
#       return_references=[wvcq.QueryReference(link_on="hasLeadOrganisation"),wvcq.QueryReference(link_on="hasParticipantOrganisation")],
#       return_metadata=wvcq.MetadataQuery(score=True)
#   )
  if batch > limit:
    batch = limit
  
  if isKeyword:
    collection = col.query.bm25(
        query=concept,
        limit=batch,
        return_references=[wvcq.QueryReference(link_on="hasLeadOrganisation"),wvcq.QueryReference(link_on="hasParticipantOrganisation")],
        return_metadata=wvcq.MetadataQuery(score=True),
        # filters=wvcq.Filter.by_property("startDate").greater_than(
        #       datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
        #   ),
        offset=offset
    )
  else:
    collection = col.query.hybrid(
      query=concept,
      alpha=0.05,
      fusion_type=wvcq.HybridFusion.RELATIVE_SCORE,
      auto_limit=100,
      limit=batch,
      return_references=[wvcq.QueryReference(link_on="hasLeadOrganisation"),wvcq.QueryReference(link_on="hasParticipantOrganisation")],
      return_metadata=wvcq.MetadataQuery(score=True),
      # filters=wvcq.Filter.by_property("startDate").greater_than(
      #         datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
      #     ),
      offset=offset

  )

  result = [
      {
          **obj.properties,
          **{
              "uuid" : ( str(obj.uuid) ),
              "Lead Organisation": (
                  obj.references["hasLeadOrganisation"].objects[0].properties["name"]
                  if obj.references and "hasLeadOrganisation" in obj.references
                  else (
                      obj.properties["participants"][0]
                      if "participants" in obj.properties and obj.properties["participants"]
                      else ""
                  )
              ),
              "Participant Organisation(s)": (
                  ", ".join([x.properties["name"] for x in obj.references["hasParticipantOrganisation"].objects])
                  if obj.references and "hasParticipantOrganisation" in obj.references
                  else (
                      ", ".join([x for x in  obj.properties["participants"]])
                      if "participants" in obj.properties and obj.properties["participants"]
                      else ""
                  )
              ),
              "score" : (obj.metadata.score), 
              "metadata" : (obj.metadata), 
            #   "explain_score" : ( obj.metadata.explain_score ),
          },
      }
      for obj in collection.objects
  ]
  
  return result

def grab_df(concept, isKeyword=True,conceptHeading=[],limit=10000,batch=10000,offset=0):
  # print(f"start {concept}")
#   seconds = time.time()
  # client = weaviate.connect_to_local(host="10.68.169.67", port=8080)
  
  # VM
  client = weaviate.connect_to_local(host="10.68.169.51",additional_config=weaviate.config.AdditionalConfig(timeout=(30, 3000)))
  # Shiny sever
  # client = weaviate.connect_to_local(host="10.68.169.195",additional_config=weaviate.config.AdditionalConfig(timeout=(30, 3000)))
  # client = weaviate.connect_to_local(host="10.68.169.150",additional_config=weaviate.config.AdditionalConfig(timeout=(30, 3000)))


  result = grab_collection(client,concept,isKeyword,limit,batch,offset)
  while(len(result) == offset+batch) and (len(result) < (limit)):
    offset=offset+batch
    result = result + grab_collection(client,concept,isKeyword,limit,batch,offset)
    print(f"len={len(result)} offset={offset} limit={limit} batch={batch}")
    print(f"{(len(result) == batch)} and {(len(result) < (limit-offset))}")
  
  client.close()
  time.sleep(0.01)

  # return _gtr_objects_to_documents(relevant_projects.objects)
  final = pd.DataFrame.from_records(result).drop_duplicates(subset='uuid', keep="last") #.query("score > 0.05",)
  if len(final) > 0:
    final["startDate"] =  pd.to_datetime(final["startDate"], errors = 'coerce').dt.tz_convert(None)
    final["endDate"] =   pd.to_datetime(final["endDate"], errors = 'coerce').dt.tz_convert(None)
    final["concept"] = concept
    for x in conceptHeading:
      final[x] = concept
    return final
  

pd.options.display.max_columns = None

# %%
# Grab a collection of data based on either keywords or semantic search
import pandas as pd
import weaviate
import weaviate.classes as wvc
import weaviate.classes.query as wvcq
from weaviate.exceptions import WeaviateQueryError


import time
from dateutil import parser
from datetime import datetime, timezone

def grab_object_by_id(project_uuids,client=None):
  # print(f"start {concept}")
#   seconds = time.time()
  # client = weaviate.connect_to_local(host="10.68.169.67", port=8080)

  if type(project_uuids) != type([]) :
    project_uuids = [project_uuids]

  if client == None:
    close_client = True
    client = weaviate.connect_to_local(host="10.68.169.51",additional_config=weaviate.config.AdditionalConfig(timeout=(30, 3000)))
  else:
    close_client = False
  
  col = client.collections.get("Universal_project")
  print(project_uuids)

  collection = col.query.fetch_objects(
    filters=wvcq.Filter.by_id().contains_any(project_uuids),
    return_references=[wvcq.QueryReference(link_on="hasLeadOrganisation"),wvcq.QueryReference(link_on="hasParticipantOrganisation")],
    # filters=wvcq.Filter.by_property("startDate").greater_than(
    #       datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    #   ),
        )
  

  result = [
      {
          **obj.properties,
          **{
              "uuid" : ( str(obj.uuid) ),
              "Lead Organisation": (
                  obj.references["hasLeadOrganisation"].objects[0].properties["name"]
                  if obj.references and "hasLeadOrganisation" in obj.references
                  else (
                      obj.properties["participants"][0]
                      if "participants" in obj.properties and obj.properties["participants"]
                      else ""
                  )
              ),
              "Participant Organisation(s)": (
                  ", ".join([x.properties["name"] for x in obj.references["hasParticipantOrganisation"].objects])
                  if obj.references and "hasParticipantOrganisation" in obj.references
                  else (
                      ", ".join([x for x in  obj.properties["participants"]])
                      if "participants" in obj.properties and obj.properties["participants"]
                      else ""
                  )
              ),
              "score" : (obj.metadata.score), 
              "metadata" : (obj.metadata), 
            #   "explain_score" : ( obj.metadata.explain_score ),
          },
      }
      for obj in collection.objects
  ]
  if close_client: 
    client.close()
  return pd.DataFrame.from_records(result)

# %%



import pandas as pd
import weaviate
import time

client = weaviate.connect_to_local(host="10.68.169.155",additional_config=weaviate.config.AdditionalConfig(timeout=(30, 3000)))
# df = grab_df("cd8250db-5d86-4b8e-bc1b-d86454dfe241")
df = grab_object_by_id("56ae28ec-f426-43d4-9436-7ecd2f31892f",client=client)
df.head()

# %%
abstract = df.description[0]



# %%
import re

# Splitting the essay on '.', '?', and '!'
single_sentences_list = re.split(r'(?<=[.?!])\s+', abstract)
print (f"{len(single_sentences_list)} senteneces were found")

# %%
sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list)]
sentences[:3]


# %%
def combine_sentences(sentences, buffer_size=1):
    # Go through each sentence dict
    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ''

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]['sentence'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['sentence']

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += ' ' + sentences[j]['sentence']

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]['combined_sentence'] = combined_sentence

    return sentences

sentences = combine_sentences(sentences)

# %%
sentences[:3]

# %%
from langchain.embeddings import OpenAIEmbeddings
oaiembeds = OpenAIEmbeddings()


# %%
embeddings = oaiembeds.embed_documents([x['combined_sentence'] for x in sentences])


# %%
for i, sentence in enumerate(sentences):
    sentence['combined_sentence_embedding'] = embeddings[i]

# %%
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        
        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences

# %%
distances, sentences = calculate_cosine_distances(sentences)


# %%
distances[:3]


# %%
import matplotlib.pyplot as plt

plt.plot(distances)

# %%
import plotly.express as px 
px.line(y=distances)


