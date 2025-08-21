# Doccano Installation and Usage Instructions

## Installation

To Install Doccano, follow the instructions listed here: https://github.com/doccano/doccano

The easiest one is to just install through pip:
``pip install doccano``

After installation, run the following commands to initialize the doccano system:

```bash
# Initialize database.
doccano init
# Create a super user with username admin and password pass
doccano createuser --username admin --password pass
```

Then, start the doccano webserver with:

```bash
# Start a web server.
doccano webserver --port 8000
```

And run this in a separate terminal:

```bash
# Start the task queue to handle file upload/download.
doccano task
```

Now Doccano is running on http://127.0.0.1:8000/.

## Usage

On http://127.0.0.1:8000/, you will see this screen:

![1722983563536](image/DoccanoInstallationandUsageInstructions/1722983563536.png)

To annotate the course documents, first login using the account you created when installing doccano (the command above created an account with username `admin` and password `pass`).

Then create a new project by hitting the create button:

![1722983989593](image/DoccanoInstallationandUsageInstructions/1722983989593.png)

Concept Labeling is a Sequence Labeling task, so select Sequence Labeling.

![1722984295735](image/DoccanoInstallationandUsageInstructions/1722984295735.png)

Scroll down and give the project a name and ensure that the checkboxes are exactly as in this picture.

![1722984551835](image/DoccanoInstallationandUsageInstructions/1722984551835.png)

And then hit create. You should see the screen below. First, go over to the labels tab on the left.

![1722999348645](image/DoccanoInstallationandUsageInstructions/1722999348645.png)

![1722999387726](image/DoccanoInstallationandUsageInstructions/1722999387726.png)

Hit actions, then import labels. 

![1722999599781](image/DoccanoInstallationandUsageInstructions/1722999599781.png)

Click File input and select the `label_config.json` file provided. Then hit import. Alternatively, copy the label below exactly and hit save.

![1722999430474](image/DoccanoInstallationandUsageInstructions/1722999430474.png)

Now, go to dataset. Use the Actions drop down menu and hit Import Dataset. 

![1722999773470](image/DoccanoInstallationandUsageInstructions/1722999773470.png)

Click File format, and then TextFile.

![1723000797920](image/DoccanoInstallationandUsageInstructions/1723000797920.png)

And then click Drop Files here to initiate a popup window to select the presentation text files you want to be annotated.

![1723001010414](image/DoccanoInstallationandUsageInstructions/1723001010414.png)

 Wait for all of the uploads to complete, and then scroll down and hit the import button. You will be transported back to the Dataset tab.

![1723001083132](image/DoccanoInstallationandUsageInstructions/1723001083132.png)

The files are ready for annotation.

## Annotation

On the file you wish to annotate, click annotate. 

![1723001143124](image/DoccanoInstallationandUsageInstructions/1723001143124.png)

To mark a portion of text as a concept, you can either highlight it and click the concept label on the dropdown 

![1723004028019](image/DoccanoInstallationandUsageInstructions/1723004028019.png)

Or press c and then highlight:

![1723004122165](image/DoccanoInstallationandUsageInstructions/1723004122165.png)

When you are done annotation, hit datasets, and then go to `Actions > Export Dataset`

![1723004243376](image/DoccanoInstallationandUsageInstructions/1723004243376.png)

And then select the JSONL file format to export.

![1723004313193](image/DoccanoInstallationandUsageInstructions/1723004313193.png)

Then hit export. You will get a zip file containing a JSON file with all of the annotations called `all.json`.

This format will then be converted into the format we will use to train the model.
