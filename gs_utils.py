from string import ascii_uppercase
import pandas as pd


def make_gs_request(gc, spreadsheet_id, request):
    body = {
        'requests': [request]
    }

    gc.service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id, body=body).execute()


def add_conditional_formatting(gc, spreadsheet_id, grid_range):
    request = {
          'addConditionalFormatRule': {
            'rule': {
              'ranges': [
                grid_range
              ],
              'gradientRule': {
                'minpoint': {
                  'color': {
                    'red': 87/255.,
                    'green': 187/255.,
                    'blue': 38/255.
                  },
                  'type': 'MIN'
                },
                'maxpoint': {
                  'color': {
                    'red': 255/255.,
                    'green': 255/255.,
                    'blue': 255/255.
                  },
                  'type': 'MAX'
                },
              }
            },
            'index': 0
          }
        }

    make_gs_request(gc, spreadsheet_id, request)


def sort_by_column(gc, spreadsheet_id, grid_range):
    request = {
          'sortRange': {
            'range': grid_range,
            'sortSpecs': [
              {
                'dimensionIndex': 1,
                'sortOrder': 'DESCENDING'
              }
            ]
          }
        }

    make_gs_request(gc, spreadsheet_id, request)


def add_protect_range(gc, spreadsheet_id, grid_range):
    request = {
          'addProtectedRange': {
            'protectedRange': {
              'range': grid_range,
              'warningOnly': True
            }
          }
        }

    make_gs_request(gc, spreadsheet_id, request)


def add_decimal_formatting(gc, spreadsheet_id, grid_range):
    request = {
          'repeatCell': {
            'range': grid_range,
            'cell': {
              'userEnteredFormat': {
                'numberFormat': {
                  'type': 'NUMBER',
                  'pattern': '0.000'
                }
              }
            },
            'fields': 'userEnteredFormat.numberFormat'
          }
        }

    make_gs_request(gc, spreadsheet_id, request)


def add_filter(gc, spreadsheet_id, grid_range):
    request = {
        'setBasicFilter': {
          'filter': {
            'range': grid_range
            }
        }
    }

    make_gs_request(gc, spreadsheet_id, request)


# doesn't take grid range?
def freeze_columns(gc, spreadsheet_id, grid_range):
    request = {
        'updateSheetProperties': {
             'fields': 'gridProperties(frozenRowCount, frozenColumnCount)',
             'properties': {
                'gridProperties': {
                    'frozenRowCount': 1,
                    'frozenColumnCount': 2},
                'sheetId': grid_range['sheet_id']
                }
            }
    }

    make_gs_request(gc, spreadsheet_id, request)


def add_background_color(gc, spreadsheet_id, grid_range):
    request = {
      'repeatCell': {
        'range': grid_range,
        'cell': {
          'userEnteredFormat': {
            'backgroundColor': {
              'red': 50,
              'green': 50,
              'blue': 50
            },
          }
        },
        'fields': 'userEnteredFormat(backgroundColor)'
      }
    }

    make_gs_request(gc, spreadsheet_id, request)


def get_data(gc, spreadsheet_id, grid_range, header=True):
    values = gc.service.spreadsheets().values()\
        .get(spreadsheetId=spreadsheet_id, range=grid_range)\
        .execute()['values']

    if header:
        return pd.DataFrame(values[1:], columns=values[0])
    else:
        return pd.DataFrame(values)


def append_df_to_table(gc, spreadsheet_id, grid_range, df):
    body = {
      'values': df.values.tolist()
    }

    gc.service.spreadsheets().values()\
        .append(spreadsheetId=spreadsheet_id, range=grid_range,
                valueInputOption='USER_ENTERED', body=body).execute()


def write_df_to_cell(gc, spreadsheet_id, grid_range, df):
    body = {
      'values': df.values.tolist()
    }

    gc.service.spreadsheets().values()\
        .update(spreadsheetId=spreadsheet_id, range=grid_range,
                valueInputOption='USER_ENTERED', body=body).execute()
