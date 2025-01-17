// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Generated with Bot Builder V4 SDK Template for Visual Studio CoreBot v4.12.2


using Microsoft.Bot.Builder;
using Microsoft.Bot.Builder.Dialogs;
using Microsoft.Bot.Schema;
using Microsoft.Extensions.Logging;
using Microsoft.Recognizers.Text.DataTypes.TimexExpression;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace IndustrySafetyChatbot.Dialogs
{
    public class MainDialog : ComponentDialog
    {
        private readonly IncidentRecognizer _luisRecognizer;
        protected readonly ILogger Logger;

        // Dependency injection uses this constructor to instantiate MainDialog
        public MainDialog(IncidentRecognizer luisRecognizer, IncidentDialog incidentDialog, ILogger<MainDialog> logger)
            : base(nameof(MainDialog))
        {
            _luisRecognizer = luisRecognizer;
            Logger = logger;

            AddDialog(new TextPrompt(nameof(TextPrompt)));
            AddDialog(incidentDialog);
            AddDialog(new WaterfallDialog(nameof(WaterfallDialog), new WaterfallStep[]
            {
                IntroStepAsync,
                ActStepAsync,
                FinalStepAsync,
            }));

            InitialDialogId = nameof(WaterfallDialog);
        }

        private async Task<DialogTurnResult> IntroStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
           
            return await stepContext.NextAsync(null, cancellationToken);}

        private async Task<DialogTurnResult> ActStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {

            try
            {
                
                return await stepContext.BeginDialogAsync(nameof(IncidentDialog), new IncidentDetails(), cancellationToken);
               

            }
            catch (Exception ex)
            {


            }
            


            return await stepContext.NextAsync(null, cancellationToken);
        }

        private async Task<DialogTurnResult> FinalStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            string responseText = "";
           
            if (stepContext.Result is IncidentDetails result)
            {
                responseText = result.Level;
                if (!result.Level.Contains("Sorry"))
                {
                    responseText = "Incident Level is:" + result.Level + ". Please take appropriate action immeidately according to the incident level.";
                }
                responseText += "\n Hope this interaction was helpful.";

                result.Response = responseText;
                var responseMessage = ActivityFactory.FromObject(result.Response);
                await stepContext.Context.SendActivityAsync(responseMessage, cancellationToken);
            }

            // Restart the main dialog with a different message the second time around
            var promptMessage = "What else can I do for you?";
            return await stepContext.ReplaceDialogAsync(InitialDialogId, promptMessage, cancellationToken);
        }
    }
}
