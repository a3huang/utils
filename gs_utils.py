from string import ascii_uppercase
import pandas as pd


def convert_to_grid_range(spreadsheet, s):
    d = dict()
    sheet_name, cell_range = s.split('!')

    d['sheetId'] = spreadsheet.worksheet_by_title(sheet_name).id

    start, end = cell_range.split(':')

    if len(start) == 2:
        d['startRowIndex'] = int(start[1]) - 1
    d['startColumnIndex'] = ascii_uppercase.index(start[0])

    if len(end) == 2:
        d['endRowIndex'] = int(end[1])
    d['endColumnIndex'] = ascii_uppercase.index(end[0]) + 1

    return d


def make_gs_request(gc, spreadsheet_id, request):
    body = {
        'requests': [request]
    }

    gc.service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id, body=body).execute()


def add_conditional_formatting(gc, spreadsheet_id, range):
    request = {
          'addConditionalFormatRule': {
            'rule': {
              'ranges': [
                convert_to_grid_range(range)
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


def remove_conditional_formatting(gc, spreadsheet_id, sheet_id):
    request = {
      'deleteConditionalFormatRule': {
        'sheetId': sheet_id,
        'index': 0
      }
    }

    make_gs_request(gc, spreadsheet_id, request)


def sort_by_column(gc, spreadsheet_id, range):
    request = {
          'sortRange': {
            'range': convert_to_grid_range(range),
            'sortSpecs': [
              {
                'dimensionIndex': 1,
                'sortOrder': 'DESCENDING'
              }
            ]
          }
        }

    make_gs_request(gc, spreadsheet_id, request)


def add_protect_range(gc, spreadsheet_id, range):
    request = {
          'addProtectedRange': {
            'protectedRange': {
              'range': convert_to_grid_range(range),
              'warningOnly': True
            }
          }
        }

    make_gs_request(gc, spreadsheet_id, request)


def add_decimal_formatting(gc, spreadsheet_id, range):
    request = {
          'repeatCell': {
            'range': convert_to_grid_range(range),
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


def add_filter(gc, spreadsheet_id, range):
    request = {
        'setBasicFilter': {
          'filter': {
            'range': convert_to_grid_range(range)
            }
        }
    }

    make_gs_request(gc, spreadsheet_id, request)


def freeze_rows_and_columns(gc, spreadsheet_id, sheet_id, num_rows,
                            num_cols):
    request = {
        'updateSheetProperties': {
             'fields': 'gridProperties(frozenRowCount, frozenColumnCount)',
             'properties': {
                'gridProperties': {
                    'frozenRowCount': num_rows,
                    'frozenColumnCount': num_cols},
                'sheetId': sheet_id
                }
            }
    }

    make_gs_request(gc, spreadsheet_id, request)


def add_background_color(gc, spreadsheet_id, range):
    request = {
      'repeatCell': {
        'range': convert_to_grid_range(range),
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


def create_new_sheet(client, name):
    existing = [i['name'] for i in client.list_ssheets()
                if i['name'] == name]
    if len(existing) == 0:
        return client.create(name)
    else:
        raise Exception


def paste_df_in_google_sheet(spreadsheet, df, sheet='Sheet1', cell='A1'):
    s = spreadsheet.worksheet_by_title(sheet)
    s.set_dataframe(df.astype(str), cell)


def get_freeze_request(sheet_id, num_rows, num_cols):
    request = {
        'updateSheetProperties': {
             'fields': 'gridProperties(frozenRowCount, frozenColumnCount)',
             'properties': {
                'gridProperties': {
                    'frozenRowCount': num_rows,
                    'frozenColumnCount': num_cols
                },
                'sheetId': sheet_id
             }
        }
    }
    return request


def get_filter_request(sheet_id):
    request = {
        'setBasicFilter': {
            'filter': {
                'range': {
                    'sheetId': sheet_id,
                    'startColumnIndex': 0
                }
            }
        }
    }
    return request


def get_color_dict():
    d = {'grey': {'red': 50, 'green': 50, 'blue': 50},
         'green': {'red': 72, 'green': 30, 'blue': 50},
         'red': {'red': 11, 'green': 56, 'blue': 60},
         'yellow': {'red': 3, 'green': 23, 'blue': 77}
         }
    return d


def get_color_range_request(sheet_id, start_row=None, end_row=None,
                            start_col=None, end_col=None, color='grey'):
    c = get_color_dict()[color]

    request = {
      'repeatCell': {
        'range': {
          'sheetId': sheet_id,
          'startRowIndex': start_row,
          'endRowIndex': end_row,
          'startColumnIndex': start_col,
          'endColumnIndex': end_col
        },
        'cell': {
          'userEnteredFormat': {
            'backgroundColor': c,
          }
        },
        'fields': 'userEnteredFormat(backgroundColor)'
      }
    }
    return request


def get_resize_request(sheet_id, start, end):
    request = {
      'autoResizeDimensions': {
        'dimensions': {
          'sheetId': sheet_id,
          'dimension': 'COLUMNS',
          'startIndex': start,
          'endIndex': end
        }
      }
    }
    return request


def create_value_object(x):
    return {'userEnteredValue': str(x)}


def get_conditional_formatting_request(sheet_id, start, end, type,
                                       values, color):
    c = get_color_dict()[color]

    request = {
      'addConditionalFormatRule': {
        'rule': {
          'ranges': [
            {
              'sheetId': sheet_id,
              'startColumnIndex': start,
              'endColumnIndex': end,
            }
          ],
          'booleanRule': {
            'condition': {
              'type': type,
              'values': [create_value_object(i) for i in values]
            },
            'format': {
              'backgroundColor': c
            }
          }
        },
        'index': 0
      }
    }
    return request
