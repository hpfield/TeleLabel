import pandas as pd
import weaviate
import weaviate.classes as wvc
import weaviate.classes.query as wvcq
from weaviate.exceptions import WeaviateQueryError


import time
from dateutil import parser
from datetime import datetime, timezone

import re
def extract_field(df,col,drop=False):
    key_list = []
    for row in df[col].values:
        # print(row)
        if row:
            for k in row.keys():
                if not k in key_list:
                    key_list.append(k)


    def extract_single_field(x,field):
        # print(f"{field} {len(field)} {type(x)}")
        if x is not None:
          if field in x.keys():
              try:
                return float(x[field])
                
              except:
                return x[field]

        return None

    for x in key_list:
        df[f"{col}_{x}"] = df[col].apply(lambda s: extract_single_field(s,x))

    if drop:
      df.drop(columns=[col],inplace=True)


def extract_fields(df,col,key=None,drop=False):
    key_list = []
    max_len = 0
    for row in df[col].values:
        if row:
            if key:
                row = row[key]
            if max_len < len(row):
                max_len = len(row)
            for x in row:
                for k in x.keys():
                    if not k in key_list:
                        key_list.append(k)


    def extract_single_field(x,field,idx):
        # print(f"{field} {idx} {len(field)} {type(x)}")
        if x is not None:
            if len(x) > idx:
                if field in x[idx].keys():
                    return x[idx][field]

        return None

    for y in range(max_len):
        for x in key_list:
            df[f"{col}_{x}_{y}"] = df[col].apply(lambda s: extract_single_field(s,x,y))
    
    if drop:
      df.drop(columns=[col],inplace=True)

def extract_vector(df,col,key=None,drop=False):
    df[f"{col}_x"] = df[col].apply(lambda s: s[key][0])
    df[f"{col}_y"] = df[col].apply(lambda s: s[key][1])
    
    if drop:
      df.drop(columns=[col],inplace=True)


def extract_vector_array(df,col,drop=False):
    for x in range(len(col[0])):
        print(x)
        df[f"{col}_{x}"] = df[col].apply(lambda s: s[x])
    
    if drop:
      df.drop(columns=[col],inplace=True)


def write_xls(df,name):
    writer = pd.ExcelWriter(
        name,
        engine="xlsxwriter",
        datetime_format="mmm d yyyy hh:mm:ss",
        date_format="mmmm dd yyyy",
    )

    df.to_excel(writer, sheet_name="Sheet1")


    # import plotly.express as px
    # fig = px.scatter(df,x="_additional_featureProjection_x",y="_additional_featureProjection_y",color="_additional_distance",hover_data=[ "title"])
    # fig.show()

    # Get the xlsxwriter workbook and worksheet objects in order
    # to set the column widths and make the dates clearer.
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]

    # Get the dimensions of the dataframe.
    (max_row, max_col) = df.shape

    # Set the column widths, to make the dates clearer.
    worksheet.set_column(1, max_col, 20)

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()


def grab_collection(client,concept,isKeyword,limit,batch,offset):
    """
    Grab a collection of data based on either keywords or semantic search
    :param client: weaviate client
    :param concept: keyword or semantic search
    :param isKeyword: boolean
    :param limit: number of items to grab
    :param batch: number of items to grab in one go
    :param offset: offset
    
    :return: list of objects
    """
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


def grab_df(
        concept, isKeyword=True, conceptHeading=[], limit=10000, batch=10000, offset=0
):
    """
    Wrapper function to grab a collection of data based on either keywords or semantic search and 
    return a pandas dataframe.

    :param concept: keyword or semantic search
    :param isKeyword: boolean
    :param conceptHeading: list of strings
    :param limit: number of items to grab
    :param batch: number of items to grab in one go
    :param offset: offset

    :return: pandas dataframe
    """
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
  

def grab_object_by_id(
    project_uuids, client=None
):
    """
    Grab an object by its id.

    :param project_uuids: list of project uuids
    :param client: weaviate client
    :return: pandas dataframe
    """
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

if __name__ == "__main__":
    client = weaviate.connect_to_local(host="10.68.169.51",additional_config=weaviate.config.AdditionalConfig(timeout=(30, 3000)))
    # df = grab_df("cd8250db-5d86-4b8e-bc1b-d86454dfe241")
    df = grab_object_by_id("56ae28ec-f426-43d4-9436-7ecd2f31892f",client=client)
    df.head()