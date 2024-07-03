import pandas as pd
from pandas import DataFrame
import requests
import numpy as np


# Extract data from source. format and organize data make it ready for the transformation phase
class DataExtractor:
    data: DataFrame
    source: str
    input_path_or_url: str
    output_path: str
    step_counter: int

    def __init__(
        self,
        input_path_or_url: str,
        output_path: str,
        source="http",
        source_credentials_path=None,
    ):
        self.data = DataFrame()
        self.source = source
        self.input_path_or_url = input_path_or_url
        self.output_path = output_path
        self.step_counter = 0

    def _check_step_counter(self, step_number: int):
        if self.step_counter != step_number:
            return None

    def _increment_step_counter(self):
        self.step_counter = self.step_counter + 1

    def get_dataset(self):
        return self.data

    def set_dataset(self, data: DataFrame):
        self.data = data

    # TODO
    def display_columns():
        pass

    def download_dataset(self):
        if self.source == "http":
            response = requests.get(self.input_path_or_url)
            if response.status_code == 200:
                # Write the file
                with open(self.output_path, "wb") as output_file:
                    output_file.write(response.content)
                    output_file.close()
                    # Update step counter
            else:
                # Raise a error
                raise Exception("Remote Data Source with Error")
        # elif self.data_source == "bigquery":
        #     # Use google api library
        #     pass
        # elif self.data_source == "aws":
        #     # Use boto3
        #     pass
        # elif self.data_source == "ftp":
        #     # Use ftp
        #     pass
        else:
            raise Exception("Remote Data Source with Error")

    def read_dataset(self, file_format="csv"):
        # Check for step counter

        if file_format == "csv":
            self.data = pd.read_csv(self.output_path)
        if file_format == "excel":
            self.data = pd.read_excel(self.output_path)

        # Update step counter

    # Columns Operations
    def format_columns(self) -> None:
        # Check if data is not empty
        if self.data is None:
            raise Exception("Data do not exist")

        # TODO: Make rename interactive
        # TODO: Set columns data types interactive
        new_names = {}
        new_dtypes = {}
        valid_dtypes = ("str", "float", "int", "bool")
        valid_name_formats = ("", "\n", "\t")
        is_complete = False

        print("################## COLUMNS OLD NAMES ############################")
        print(self.data.columns)
        print("#########################################################")
        input("Press any key to continue... \n")

        while not is_complete:
            for column in self.data.columns:
                print(f"################## {column} ############################")
                print(f"##### Column name to be change: {column}")
                print(f"#####           with new dtype: {self.data[column].dtypes}")

                new_dtype = input(
                    "##### Input new column dtype (leave empty to keep old dtype): "
                )
                if new_dtype in valid_dtypes:
                    new_dtypes.update({column: new_dtype})
                else:
                    print(
                        f"##### Skiping new dtype and keeping old dtype: {self.data[column].dtypes}"
                    )
                    print("##### New dtype do not match with valids dtypes")

                new_name = input(
                    "##### Input new column name (leave empty to keep old name): "
                )
                if not new_name in valid_name_formats:
                    new_names.update({column: new_name})
                else:
                    print(f"##### Skiping new name for {column}, keeping old name")
                    print("##### New name do not match with valids names formats")

                print("######################################################### \n")

            print(new_names)
            print(new_dtypes)
            option = input(
                "##### Are you sure the new names are corrects (y/n) (default: N): "
            )
            option = option.upper()
            if option == "Y":
                self.data.astype(new_dtypes, copy=False)
                self.data.rename(columns=new_names, inplace=True)
                is_complete = True
            elif option == "N":
                print("##### Trying again")
                new_names = {}
                new_dtypes = {}
            else:
                print("##### Invalid input, exiting")
                is_complete = True

            print("##### Columns successfuly formated")

    def create_columns(self):
        is_complete = False
        while not is_complete:
            print(self.data.columns)
            option = input(
                "##### Do you want to add an new column? (y/n) (default: y): "
            )
            option = option.upper()
            if option == "Y":
                column_name = str(input())
                column_position = int(input())
                # TODO add dtype definition
                self.data.insert(
                    column_position,
                    column_name,
                    np.empty(self.data.shape[0], dtype=np.str_),
                )
            elif option == "N":
                print("##### No columns created, exiting")
                is_complete = True
            else:
                print("##### Invalid input, exiting")
                is_complete = True

        print("##### Columns successfuly created")

    def delete_columns(self):
        columns_to_delete = []
        is_complete = False
        print(self.data.columns)
        option = input("##### Do you want to DELETE a column? (y/n) (default: y): ")
        option = option.upper()
        while not is_complete:
            if option == "Y":
                column_name = str(input("Name of the column to be deleted: "))
                if column_name in self.data.columns:
                    columns_to_delete.append(column_name)
                else:
                    print("Name not exist, try again")
                option = input(
                    "##### Do you want to DELETE AN OTHER column? (y/n) (default: y): "
                )
                option = option.upper()
            else:
                is_complete = True

        self.data.drop(columns=columns_to_delete)
        print("##### Columns successfuly DELETED")
