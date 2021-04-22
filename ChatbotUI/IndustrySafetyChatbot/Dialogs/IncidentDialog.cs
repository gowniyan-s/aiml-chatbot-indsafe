// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Generated with Bot Builder V4 SDK Template for Visual Studio CoreBot v4.12.2

using Microsoft.Bot.Builder;
using Microsoft.Bot.Builder.Dialogs;
using Microsoft.Bot.Schema;
using Microsoft.Recognizers.Text.DataTypes.TimexExpression;
using System.Threading;
using System.Threading.Tasks;
using System.Net;
using System.IO;

namespace IndustrySafetyChatbot.Dialogs
{
    public class IncidentDialog : CancelAndHelpDialog
    {
        private const string DestinationStepMsgText = "Please describe the incident for me to help you";
        private const string OriginStepMsgText = "Sorry I dont have the information regarding the same";

        public IncidentDialog()
            : base(nameof(IncidentDialog))
        {
            AddDialog(new TextPrompt(nameof(TextPrompt)));
            AddDialog(new ConfirmPrompt(nameof(ConfirmPrompt)));
            AddDialog(new DateResolverDialog());
            AddDialog(new WaterfallDialog(nameof(WaterfallDialog), new WaterfallStep[]
            {
                DestinationStepAsync,
                ConfirmStepAsync,
                FinalStepAsync,
            }));

            // The initial child Dialog to run.
            InitialDialogId = nameof(WaterfallDialog);
        }

        private async Task<DialogTurnResult> DestinationStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            var incidentDetails = (IncidentDetails)stepContext.Options;
            try
            {
                

                if (incidentDetails.Incident == null)
                {
                    var promptMessage = MessageFactory.Text(DestinationStepMsgText, DestinationStepMsgText, InputHints.ExpectingInput);
                    return await stepContext.PromptAsync(nameof(TextPrompt), new PromptOptions { Prompt = promptMessage }, cancellationToken);
                }

            }
            catch (System.Exception ex)
            {

                
            }
            

            return await stepContext.NextAsync(incidentDetails.Incident, cancellationToken);
        }


        private async Task<DialogTurnResult> ConfirmStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            var incidentDetails = (IncidentDetails)stepContext.Options;

            string accidentDescription = (string)stepContext.Result;
            incidentDetails.Incident = accidentDescription;

            var messageText = $"Please confirm, the incident your reproting is : {accidentDescription} . Is this correct?";
            var promptMessage = MessageFactory.Text(messageText, messageText, InputHints.ExpectingInput);

            return await stepContext.PromptAsync(nameof(ConfirmPrompt), new PromptOptions { Prompt = promptMessage}, cancellationToken);
        }

        private async Task<DialogTurnResult> FinalStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            if ((bool)stepContext.Result)
            {
                var incidentDetails = (IncidentDetails)stepContext.Options;
                incidentDetails.Level = GetResponsefromModel(incidentDetails.Incident);
                return await stepContext.EndDialogAsync(incidentDetails, cancellationToken);
            }

            return await stepContext.EndDialogAsync(null, cancellationToken);
        }

        private static bool IsAmbiguous(string timex)
        {
            var timexProperty = new TimexProperty(timex);
            return !timexProperty.Types.Contains(Constants.TimexTypes.Definite);
        }

        private string GetResponsefromModel(string query) 
        {
            string html = "Sorry i dont have information on the same.";
            try
            {
                if (query != "")
                {

                    string url = @"http://127.0.0.1:5000?query="+ query;

                    HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url);
                    request.AutomaticDecompression = DecompressionMethods.GZip;

                    using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                    using (Stream stream = response.GetResponseStream())
                    using (StreamReader reader = new StreamReader(stream))
                    {
                        html = reader.ReadToEnd();
                    }
                }

            }
            catch (System.Exception)
            {

                //
            }
            return html;
        }
    }
}
