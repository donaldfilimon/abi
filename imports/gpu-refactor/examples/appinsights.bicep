// Simple App Insights resource for a web app in the same RG as the template
// Requires: Azure CLI, contributor role on resource group
param appName string

resource ai 'microsoft.insights/components@2023-05-01' = {
  name: '${appName}-ai'
  location: resourceGroup().location
  tags: {}
  properties: {
    Application_Type: 'Web'
    Request_Source: 'Bicep'
    RetentionInDays: 30
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

output appInsightsId string = ai.id
output appInsightsName string = ai.name
output appInsightsInstrumentationKey string = ai.properties.InstrumentationKey
