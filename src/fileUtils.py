from google.cloud import storage
from google.oauth2 import service_account
import os

class FileUtils:
    def __init__(self,credentials_path,bucket_name):
        self.input_folder = 'in/'
        self.output_folder = 'out/'    
        self.bucket_name = bucket_name
        cred = service_account.Credentials.from_service_account_file(credentials_path)
        self.storage_client = storage.Client(credentials=cred)
    
    def sync_in_folder(self):
        # print( 'sync IN folder')
        files_in_remote = self.get_all_files(self.input_folder)
        files_local = [self.input_folder + f for f in os.listdir(self.input_folder)]
        new_files = [f for f in files_in_remote if not f in files_local]
        # print (files_in_remote)
        # print (files_local)
        # print (new_files)
        for file_path in new_files:
            self.download_file(file_path, file_path)



    def sync_out_folder(self):
        print( 'sync OUT folder')
        files_in_remote = self.get_all_files(self.output_folder)
        files_local = [self.output_folder + f for f in os.listdir(self.output_folder)]
        new_files = [f for f in files_local if not f in files_in_remote]
        print (files_in_remote)
        print (files_local)
        print (new_files)
        for file_path in new_files:
            self.upload_file(file_path, file_path)

    def lista_archivos_no_procesados(self):
        files_list = []
        processed_file_names = []
        for file in os.listdir(self.output_folder):
            if os.path.isfile(os.path.join(self.output_folder,file)):
                processed_file_names.append(file.split('.')[0])
        
        processed_file_names = list( dict.fromkeys(processed_file_names) )
        print (processed_file_names)

        print( 'buscar archivos no procesados en IN')
        for file in os.listdir(self.input_folder):
            if os.path.isfile(os.path.join(self.input_folder,file)) and not file.split('.')[0] in processed_file_names :
                files_list.append(file)
        
        return files_list
    
    def get_all_files(self,prefix = ''):
        files = self.storage_client.list_blobs(self.bucket_name,prefix=prefix)
        files_list = []
        for f in files: 
            if f.name[-1] != '/': 
                files_list.append(f.name)
        return files_list

    def download_file(self,remote_file_path,local_destination_path):
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(remote_file_path)
        blob.download_to_filename(local_destination_path)
        print(f"Archivo '{remote_file_path}' descargado a '{local_destination_path}")

    def upload_file(self, local_file_path, destination_path):
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(destination_path)
        blob.upload_from_filename(local_file_path)
        print(f"Archivo '{local_file_path}' subido a '{destination_path}")
