import torch 
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, dataloader, optimizer, num_epochs=10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs[0]
            optimizer.zero_grad()

            outputs1, outputs2, outputs3 = model(inputs, targets)
            outputs1 = outputs1.unsqueeze(0)
            outputs2 = outputs2.unsqueeze(0)
            outputs3 = outputs3.unsqueeze(0)

            outputs1 = outputs1.float()
            outputs2 = outputs2.float()
            outputs3 = outputs3.float()

            targets1 = targets[0].squeeze(0).float().to(device)
            targets2 = targets[1].squeeze(0).float().to(device)
            targets3 = targets[2].squeeze(0).float().to(device)

            loss1 = F.mse_loss(outputs1, targets1)
            loss2 = F.mse_loss(outputs2, targets2)
            loss3 = F.mse_loss(outputs3, targets3)

            epoch_loss = loss1 + loss2 + loss3

            epoch_loss.backward()
            optimizer.step()

            total_loss += epoch_loss

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

    return average_loss




def find_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key

    return None

def decode_input(encoded_input, token_to_int_mapping):
    decoded_input=[]
    for idx in encoded_input:
      if int(idx) !=0 and find_key_by_value(token_to_int_mapping, int(idx)) is not None:
        decoded_input.append(find_key_by_value(token_to_int_mapping, int(idx)))
    return decoded_input

def inference(model,input_encoding):
    # Load the trained model
    # model = TransformerSeq2Seq(vocab_size_condition1=1000, vocab_size_condition2=1000, vocab_size_condition3=1000)
    # model_state_dict = torch.load('your_model_checkpoint.pth')
    # model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()


    # Create a placeholder tensor for target_sequence during inference
    placeholder_target = [torch.zeros((1,1,100)).long().to(device),torch.zeros((1,1,100)).long().to(device),torch.zeros((1,1,100)).long().to(device)]  # Adjust max_sequence_length accordingly7



    # Perform inference
    with torch.no_grad():
        # Pass only input_encoding and the placeholder_target to the model during inference
        output_probs_condition1, output_probs_condition2, output_probs_condition3 = model(input_encoding, targets=placeholder_target, train_mode=False)

    # Post-process output if necessary
    output_condition1 = output_probs_condition1.cpu().numpy()
    output_condition2 = output_probs_condition2.cpu().numpy()
    output_condition3 = output_probs_condition3.cpu().numpy()

    decoded_condition1 = decode_input(output_probs_condition1, token_to_int_mapping)
    decoded_condition2 = decode_input(output_probs_condition2, token_to_int_mapping)
    decoded_condition3 = decode_input(output_probs_condition3, token_to_int_mapping)

    reactant,product= separate_compounds(input_encoding)

    print("This is the original Reaction", input_encoding,'The reactant is ',reactant, 'The product is ',product)
    print("The Reagant of the original Reaction is",' '.join(decoded_condition1))
    print("The Solvent of the original Reaction is",' '.join(decoded_condition2))
    print("The Catalyst of the original Reaction is",' '.join(decoded_condition3))


    return output_condition1, output_condition2, output_condition3

