import React, { useState } from 'react';
import { View, Text, StyleSheet, SafeAreaView, TextInput, TouchableOpacity, Platform, ScrollView, Image, KeyboardAvoidingView } from 'react-native';
import { Upload, Send } from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
import * as MailComposer from 'expo-mail-composer';
import CustomTabBar from '../../components/CustomTabBar';

const problemTypes = ['Bug', 'Suggestion', 'Other'];

const ReportScreen = () => {
  const [problem, setProblem] = useState('Bug');
  const [title, setTitle] = useState('');
  const [details, setDetails] = useState('');
  const [email, setEmail] = useState('');
  const [attachment, setAttachment] = useState<string | null>(null);

  // Image picker for attachment
  const handlePickAttachment = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.7,
    });
    if (!result.canceled && result.assets && result.assets.length > 0) {
      setAttachment(result.assets[0].uri);
    }
  };

  const handleSendReport = async () => {
    if (!title || !details || !email) {
      alert('Please fill in all fields.');
      return;
    }

    const options: MailComposer.MailComposerOptions = {
      recipients: ['placeholder@example.com'], // Replace with your desired email
      subject: `[${problem}] ${title}`,
      body: `Details:\n${details}\n\nEmail: ${email}`,
      attachments: attachment ? [attachment] : [],
    };

    try {
      await MailComposer.composeAsync(options);
    } catch (e) {
      alert('Could not open mail composer.');
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: '#fafafa' }}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 24}
      >
        <SafeAreaView style={{ flex: 1 }}>
          <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
            <Text style={styles.title}>Report</Text>

            {/* Problem Type */}
            <View style={styles.card}>
              <Text style={styles.cardTitle}>What is the problem?</Text>
              {problemTypes.map((type) => (
                <TouchableOpacity
                  key={type}
                  style={styles.radioRow}
                  onPress={() => setProblem(type)}
                  activeOpacity={0.7}
                >
                  <View style={[styles.radioOuter, problem === type && styles.radioOuterActive]}>
                    {problem === type && <View style={styles.radioInner} />}
                  </View>
                  <Text style={styles.radioLabel}>{type}</Text>
                </TouchableOpacity>
              ))}
            </View>

            {/* Form Fields */}
            <View style={styles.card}>
              <Text style={styles.label}>Title</Text>
              <TextInput
                style={styles.input}
                value={title}
                onChangeText={setTitle}
                placeholder="Title"
                placeholderTextColor="#bdbdbd"
              />
              <Text style={styles.label}>Details</Text>
              <TextInput
                style={[styles.input, { height: 60 }]}
                value={details}
                onChangeText={setDetails}
                placeholder="Description"
                placeholderTextColor="#bdbdbd"
                multiline
              />

              {/* Attachment */}
              <Text style={[styles.label, { marginTop: 16 }]}>Attachment</Text>
              <View style={[styles.attachmentBox, attachment ? styles.attachmentBoxWithImage : null]}>
                {!attachment ? (
                  <TouchableOpacity style={styles.attachmentBtn} onPress={handlePickAttachment}>
                    <Upload color="#bdbdbd" size={22} style={{ marginRight: 8 }} />
                    <Text style={styles.attachmentBtnText}>Choose from album</Text>
                  </TouchableOpacity>
                ) : (
                  <View style={styles.attachmentImageWrapper}>
                    <Image
                      source={{ uri: attachment }}
                      style={styles.attachmentImage}
                      resizeMode="contain"
                    />
                    <TouchableOpacity style={styles.changeBtn} onPress={handlePickAttachment}>
                      <Text style={styles.changeBtnText}>Change</Text>
                    </TouchableOpacity>
                  </View>
                )}
              </View>

              {/* Email */}
              <Text style={[styles.label, { marginTop: 16 }]}>Email Address</Text>
              <TextInput
                style={styles.input}
                value={email}
                onChangeText={setEmail}
                placeholder="example@email.com"
                placeholderTextColor="#bdbdbd"
                keyboardType="email-address"
                autoCapitalize="none"
              />
            </View>

            {/* Send Button */}
            <TouchableOpacity style={styles.sendBtn} onPress={handleSendReport}>
              <Send color="#222" size={28} style={{ marginRight: 12 }} />
              <Text style={styles.sendBtnText}>Send Report</Text>
            </TouchableOpacity>
          </ScrollView>
        </SafeAreaView>
      </KeyboardAvoidingView>
      <CustomTabBar />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 24,
    backgroundColor: '#fafafa',
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#222',
    marginBottom: 24,
    marginTop: 8,
    textAlign: 'center',
  },
  card: {
    width: '100%',
    maxWidth: 420,
    backgroundColor: '#fff',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    padding: 20,
    marginBottom: 24,
  },
  cardTitle: {
    fontWeight: 'bold',
    color: '#bdbdbd',
    fontSize: 17,
    marginBottom: 16,
    alignSelf: 'center',
  },
  radioRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  radioOuter: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#bdbdbd',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  radioOuterActive: {
    borderColor: '#facc15',
  },
  radioInner: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#facc15',
  },
  radioLabel: {
    fontSize: 17,
    color: '#222',
  },
  label: {
    fontWeight: 'bold',
    color: '#888',
    fontSize: 15,
    marginBottom: 6,
    marginTop: 10,
  },
  input: {
    backgroundColor: '#fafafa',
    borderWidth: 1,
    borderColor: '#bdbdbd',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: Platform.OS === 'ios' ? 12 : 8,
    fontSize: 16,
    marginBottom: 8,
    color: '#222',
  },
  attachmentBox: {
    borderWidth: 1,
    borderColor: '#bdbdbd',
    borderRadius: 10,
    borderStyle: 'dashed',
    backgroundColor: '#fafafa',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 100,
    width: '100%',
    maxHeight: 220,
    marginTop: 8,
    padding: 16,
  },
  attachmentBoxWithImage: {
    borderStyle: 'solid',
    minHeight: 120,
    width: '100%',
    maxHeight: 220,
    padding: 16,
  },
  attachmentBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ededed',
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 10,
    marginTop: 8,
  },
  attachmentBtnText: {
    color: '#bdbdbd',
    fontSize: 15,
  },
  attachmentImageWrapper: {
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  attachmentImage: {
    width: '100%',
    height: 180,
    maxWidth: '100%',
    maxHeight: 180,
    borderRadius: 8,
  },
  changeBtn: {
    marginTop: 8,
    backgroundColor: '#ededed',
    borderRadius: 8,
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  changeBtnText: {
    color: '#bdbdbd',
    fontSize: 14,
  },
  sendBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fef3c7',
    borderRadius: 16,
    paddingVertical: 22,
    justifyContent: 'center',
    width: '100%',
    maxWidth: 420,
    marginTop: 8,
    marginBottom: 32,
  },
  sendBtnText: {
    color: '#222',
    fontSize: 20,
  },
});

export default ReportScreen; 