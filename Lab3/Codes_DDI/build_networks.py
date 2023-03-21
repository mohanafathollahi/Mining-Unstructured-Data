def build_network(idx, glove_embeddings_index):

    # sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()

    # less frequent than 10000 will get removed
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    inptW = Input(shape=(max_len), dtype='int32')
    embW = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                        input_length=max_len, mask_zero=False)(inptW)
    
    inptL = Input(shape=(max_len,))
    embL = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                        input_length=max_len, mask_zero=False)(inptL)
     
    inptP = Input(shape=(max_len,))
    embP = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                        input_length=max_len, mask_zero=False)(inptP)  
        
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    # inptW = Input(shape=(max_len), dtype='int32')
    # x = glove_embeddings_index(inptW)
    
    # -----------------------------------------------------------------------------------------------------------------------------------

    x = Conv1D(filters=config_filters, kernel_size=config_kernel_size,
               strides=1, activation='relu', padding='same', name="Conv1D_W")(embW)
    
    x = MaxPool1D(pool_size=(max_len - 3 + 1), strides=1, padding='valid', name='MaxPool1D_W')(x)
    
    y = Conv1D(filters=config_filters, kernel_size=config_kernel_size,
               strides=1, activation='relu', padding='same', name="Conv1D_L")(embL)
    
    y = MaxPool1D(pool_size=(max_len - 3 + 1), strides=1, padding='valid', name='MaxPool1D_L')(y)
    
    z = Conv1D(filters=config_filters, kernel_size=config_kernel_size,
               strides=1, activation='relu', padding='same', name="Conv1D_P")(embP)
   
    z = MaxPool1D(pool_size=(max_len - 3 + 1), strides=1, padding='valid', name='MaxPool1D_P')(z)
    
    w = Concatenate(axis=1, name='Concatenate_MaxPool')([x,y,z])

    w = Flatten()(w)

    w = Dense(n_labels, activation='softmax', name="Final_Dense")(w)

    model = Model(inputs=[inptW, inptL, inptP], outputs=w)
    # model = Model(inptW, outputs=x)

    # -----------------------------------------------------------------------------------------------------------------------------------

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model
    
def build_network(idx, glove_embeddings_index):

    # sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()

    # less frequent than 10000 will get removed
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    # inptW = Input(shape=(max_len,))
    # x = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
    #                     input_length=max_len, mask_zero=False)(inptW) 
        
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    inptW = Input(shape=(max_len), dtype='int32')
    embGVLC = glove_embeddings_index(inptW)
    
    # -----------------------------------------------------------------------------------------------------------------------------------

    x = Conv1D(filters=config_filters, kernel_size=config_kernel_size,
               strides=1, activation='relu', padding='same', name="Conv1D_x")(embGVLC)

    x = Flatten()(x)

    x = Dense(n_labels, activation='softmax', name="Dense_z")(x)

    # model = Model(inputs=[inptW, inptP], outputs=z)
    model = Model(inptW, outputs=x)

    # -----------------------------------------------------------------------------------------------------------------------------------

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model

def build_network(idx, glove_embeddings_index):

    # sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()

    # less frequent than 10000 will get removed
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()
    lstm_units = max_len

    # -----------------------------------------------------------------------------------------------------------------------------------
    inptW = Input(shape=(max_len,))
    x = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                  input_length=max_len, mask_zero=False)(inptW)

    # x = glove_embeddings_index(inptW)
    # -----------------------------------------------------------------------------------------------------------------------------------

    inptL = Input(shape=(max_len,))
    y = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                     input_length=max_len, mask_zero=False)(inptL)

    inptP = Input(shape=(max_len,))
    z = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                     input_length=max_len, mask_zero=False)(inptP)

    # -----------------------------------------------------------------------------------------------------------------------------------
    x = Bidirectional(LSTM(units=lstm_units, recurrent_dropout=0.1, return_sequences=True))(x)
    x = TimeDistributed(Dense(n_labels, activation="softmax"))(x)#(timeDistributed)
    # -----------------------------------------------------------------------------------------------------------------------------------
    y = LSTM(units=lstm_units, recurrent_dropout=0.1, return_sequences=True)(y)
    y = TimeDistributed(Dense(n_labels, activation="softmax"))(y)#(timeDistributed)
    # -----------------------------------------------------------------------------------------------------------------------------------
    z = LSTM(units=lstm_units, recurrent_dropout=0.1, return_sequences=True)(z)
    z = TimeDistributed(Dense(n_labels, activation="softmax"))(z)#(timeDistributed)
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    final = Concatenate(axis=1, name='Concatenate_Dense')([x,y,z])
    
    # -----------------------------------------------------------------------------------------------------------------------------------

    final = Flatten(name='Flatten')(final)

    final = Dropout(0.5)(final)
    final = Dense(n_labels, activation='softmax', name="Final_Dense")(final)
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    # final = Model(inputs=[inptW], outputs=final)
    final = Model(inputs=[inptW, inptL, inptP], outputs=final)
    # -----------------------------------------------------------------------------------------------------------------------------------

    final.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return final

def build_network(idx, glove_embeddings_index):

    # sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()

    # less frequent than 10000 will get removed
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    inptW = Input(shape=(max_len,))
    # x = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
    #               input_length=max_len, mask_zero=False)(inptW)

    x = glove_embeddings_index(inptW)
    # -----------------------------------------------------------------------------------------------------------------------------------

    inptL = Input(shape=(max_len,))
    embL = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                     input_length=max_len, mask_zero=False)(inptL)

    inptP = Input(shape=(max_len,))
    embP = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                     input_length=max_len, mask_zero=False)(inptP)

    # -----------------------------------------------------------------------------------------------------------------------------------

    lstm_x_1 = LSTM(config_hidden_dims, activation='softmax', name="LSTM_LW")(x)
    lstm_x_2 = LSTM(config_hidden_dims, activation='softmax', name="LSTM_LW")(x)

    maxpool_x_1 = MaxPool1D(pool_size=max_len + 1,
                           strides=1, padding='valid')(lstm_x_1)
    maxpool_x_2 = MaxPool1D(pool_size=max_len + 1,
                           strides=1, padding='valid')(lstm_x_2)
    # -----------------------------------------------------------------------------------------------------------------------------------
    lstm_y_1 = LSTM(config_hidden_dims, activation='softmax', name="LSTM_Lemma")(embL)
    lstm_y_2 = LSTM(config_hidden_dims, activation='softmax', name="LSTM_Lemma")(embL)

    maxpool_y_1 = MaxPool1D(pool_size=max_len + 1,
                           strides=1, padding='valid')(lstm_y_1)
    maxpool_y_2 = MaxPool1D(pool_size=max_len + 1,
                           strides=1, padding='valid')(lstm_y_2)
    # -----------------------------------------------------------------------------------------------------------------------------------
    lstm_z_1 = LSTM(config_hidden_dims, activation='softmax', name="LSTM_Pos")(embP)
    lstm_z_2 = LSTM(config_hidden_dims, activation='softmax', name="LSTM_Pos")(embP)

    maxpool_z_1 = MaxPool1D(pool_size=max_len + 1,
                           strides=1, padding='valid')(lstm_z_1)
    maxpool_z_2 = MaxPool1D(pool_size=max_len + 1,
                           strides=1, padding='valid')(lstm_z_2)
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    final = Concatenate(axis=1, name='Concatenate_Dense')([maxpool_x_1,maxpool_x_2,maxpool_y_1,maxpool_y_2,maxpool_z_1,maxpool_z_2])
    
    # -----------------------------------------------------------------------------------------------------------------------------------

    final = Flatten(name='Flatten')(final)

    final = Dropout(0.5)(final)
    final = Dense(n_labels, activation='softmax', name="Final_Dense")(final)
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    # final = Model(inputs=[inptW], outputs=final)
    final = Model(inputs=[inptW, inptL, inptP], outputs=final)
    # -----------------------------------------------------------------------------------------------------------------------------------

    final.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return final

def build_network(idx, glove_embeddings_index):

    # sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()

    # less frequent than 10000 will get removed
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()

    # -----------------------------------------------------------------------------------------------------------------------------------
    inptW = Input(shape=(max_len,))
    x = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                  input_length=max_len, mask_zero=False)(inptW)

    # inptW = Input(shape=(max_len), dtype='int32')
    # x = glove_embeddings_index(inptW)
    # -----------------------------------------------------------------------------------------------------------------------------------

    inptL = Input(shape=(max_len,))
    embL = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                     input_length=max_len, mask_zero=False)(inptL)

    inptP = Input(shape=(max_len,))
    embP = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                     input_length=max_len, mask_zero=False)(inptP)

    # -----------------------------------------------------------------------------------------------------------------------------------

    conv_0 = Conv1D(filters=config_filters, kernel_size=(3), activation='relu',
                    kernel_regularizer=regularizers.l2(config_kernel_regularizer_l2))(x)
    conv_1 = Conv1D(filters=config_filters, kernel_size=(4), activation='relu',
                    kernel_regularizer=regularizers.l2(config_kernel_regularizer_l2))(x)
    conv_2 = Conv1D(filters=config_filters, kernel_size=(5), activation='relu',
                    kernel_regularizer=regularizers.l2(config_kernel_regularizer_l2))(x)

    maxpool_0 = MaxPool1D(pool_size=(max_len - 3 + 1),
                          strides=1, padding='valid')(conv_0)
    maxpool_1 = MaxPool1D(pool_size=(max_len - 4 + 1),
                          strides=1, padding='valid')(conv_1)
    maxpool_2 = MaxPool1D(pool_size=(max_len - 5 + 1),
                          strides=1, padding='valid')(conv_2)
    # -----------------------------------------------------------------------------------------------------------------------------------

    model1 = Concatenate(axis=1, name='Concatenate_MaxPool_1')(
        [maxpool_0, maxpool_1, maxpool_2])
    model1 = Dense(n_labels, activation="softmax")(model1)
    model1 = Model([inptW], outputs=model1)

    # -----------------------------------------------------------------------------------------------------------------------------------
    conv_01 = Conv1D(filters=config_filters, kernel_size=(3), activation='relu',
                     kernel_regularizer=regularizers.l2(config_kernel_regularizer_l2))(embP)
    conv_11 = Conv1D(filters=config_filters, kernel_size=(4), activation='relu',
                     kernel_regularizer=regularizers.l2(config_kernel_regularizer_l2))(embP)
    conv_21 = Conv1D(filters=config_filters, kernel_size=(5), activation='relu',
                     kernel_regularizer=regularizers.l2(config_kernel_regularizer_l2))(embP)

    maxpool_01 = MaxPool1D(pool_size=(max_len - 3 + 1),
                           strides=1, padding='valid')(conv_01)
    maxpool_11 = MaxPool1D(pool_size=(max_len - 4 + 1),
                           strides=1, padding='valid')(conv_11)
    maxpool_21 = MaxPool1D(pool_size=(max_len - 5 + 1),
                           strides=1, padding='valid')(conv_21)

    # -----------------------------------------------------------------------------------------------------------------------------------
    model2 = Concatenate(axis=1, name='Concatenate_MaxPool_2')(
        [maxpool_01, maxpool_11, maxpool_21])
    model2 = Dense(n_labels, activation="softmax")(model2)

    model2 = Model([inptP], outputs=model2)
    # -----------------------------------------------------------------------------------------------------------------------------------

    conv_02 = Conv1D(filters=config_filters, kernel_size=(3), activation='relu',
                     kernel_regularizer=regularizers.l2(config_kernel_regularizer_l2))(embL)
    conv_12 = Conv1D(filters=config_filters, kernel_size=(4), activation='relu',
                     kernel_regularizer=regularizers.l2(config_kernel_regularizer_l2))(embL)
    conv_22 = Conv1D(filters=config_filters, kernel_size=(5), activation='relu',
                     kernel_regularizer=regularizers.l2(config_kernel_regularizer_l2))(embL)

    maxpool_02 = MaxPool1D(pool_size=(max_len - 3 + 1),
                           strides=1, padding='valid')(conv_02)
    maxpool_12 = MaxPool1D(pool_size=(max_len - 4 + 1),
                           strides=1, padding='valid')(conv_12)
    maxpool_22 = MaxPool1D(pool_size=(max_len - 5 + 1),
                           strides=1, padding='valid')(conv_22)

    # -----------------------------------------------------------------------------------------------------------------------------------
    model3 = Concatenate(axis=1, name='Concatenate_MaxPool_3')(
        [maxpool_02, maxpool_12, maxpool_22])
    model3 = Dense(n_labels, activation="softmax")(model3)

    model3 = Model([inptL], outputs=model3)

    # -----------------------------------------------------------------------------------------------------------------------------------

    final = Concatenate(axis=1, name='Concatenate_MaxPool')(
        [model1.output, model2.output, model3.output])

    # -----------------------------------------------------------------------------------------------------------------------------------

    final = Flatten(name='Flatten')(final)

    final = Dropout(0.5)(final)
    final = Dense(n_labels, activation='softmax', name="Final_Dense")(final)

    # -----------------------------------------------------------------------------------------------------------------------------------
    final = Model([model1.input, model2.input, model3.input], outputs=final)
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    final.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return final

def build_network(idx, glove_embeddings_index):

    # sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()

    # less frequent than 10000 will get removed
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    inptW = Input(shape=(max_len,))
    x = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                  input_length=max_len, mask_zero=False)(inptW)

    # x = glove_embeddings_index(inptW)
    # -----------------------------------------------------------------------------------------------------------------------------------

    inptL = Input(shape=(max_len,))
    embL = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                     input_length=max_len, mask_zero=False)(inptL)

    inptP = Input(shape=(max_len,))
    embP = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                     input_length=max_len, mask_zero=False)(inptP)

    # -----------------------------------------------------------------------------------------------------------------------------------
    x = Conv1D(filters=config_filters, kernel_size=config_kernel_size, activation=config_activation)(x)
    x = LSTM(config_hidden_dims, activation='softmax')(x)

    x = Dense(1, activation="softmax")(x)
    
    y = Conv1D(filters=config_filters, kernel_size=config_kernel_size, activation=config_activation)(embL)
    
    y = LSTM(config_hidden_dims, activation='softmax')(y)
    y = Dense(1, activation="softmax")(y)
    
    z = Conv1D(filters=config_filters, kernel_size=config_kernel_size, activation=config_activation)(embP)
    
    y = LSTM(config_hidden_dims, activation='softmax')(z)
    z = Dense(1, activation="softmax")(z)
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    final = Concatenate(axis=1, name='Concatenate_Dense')([x,y,z])
    
    # -----------------------------------------------------------------------------------------------------------------------------------


    final = Dropout(0.5)(final)
    final = Dense(1, activation='softmax', name="Final_Dense")(final)
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    final = Model(inputs=[inptW, inptL, inptP], outputs=final)
    # -----------------------------------------------------------------------------------------------------------------------------------

    final.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return final

def build_network(idx, glove_embeddings_index):

    # sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()

    # less frequent than 10000 will get removed
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    inptW = Input(shape=(max_len,))
    # x = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
    #               input_length=max_len, mask_zero=False)(inptW)

    x = glove_embeddings_index(inptW)
    # -----------------------------------------------------------------------------------------------------------------------------------

    inptL = Input(shape=(max_len,))
    embL = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                     input_length=max_len, mask_zero=False)(inptL)

    inptP = Input(shape=(max_len,))
    embP = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                     input_length=max_len, mask_zero=False)(inptP)

    # -----------------------------------------------------------------------------------------------------------------------------------

    x = Bidirectional(LSTM(config_hidden_dims, activation=config_activation, name="Bidirectional_LW"))(x)
    x = Dense(n_labels, activation="softmax")(x)
    
    y = Bidirectional(LSTM(config_hidden_dims, activation=config_activation, name="Bidirectional_Lemma"))(embL)
    y = Dense(n_labels, activation="softmax")(y)
    
    z = Bidirectional(LSTM(config_hidden_dims, activation=config_activation, name="Bidirectional_Pos"))(embP)
    z = Dense(n_labels, activation="softmax")(z)
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    final = Concatenate(axis=1, name='Concatenate_Dense')([x,y,z])
    
    # -----------------------------------------------------------------------------------------------------------------------------------

    final = Flatten(name='Flatten')(final)

    final = Dropout(0.5)(final)
    final = Dense(n_labels, activation='softmax', name="Final_Dense")(final)
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    final = Model(inputs=[inptW, inptL, inptP], outputs=final)
    # -----------------------------------------------------------------------------------------------------------------------------------

    final.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return final
